import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import com.sun.net.httpserver.*;
import java.net.http.*;

public class Main {
    static final int DIMS = 16;
    
    static class VectorItem {
        int id;
        String metadata;
        String category;
        float[] emb;
        public VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id; this.metadata = metadata; this.category = category; this.emb = emb;
        }
    }
    
    interface DistFn { float compute(float[] a, float[] b); }
    
    static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i=0; i<a.length; i++) { float d = a[i]-b[i]; s+= d*d; }
        return (float)Math.sqrt(s);
    }
    
    static float cosine(float[] a, float[] b) {
        float dot=0, na=0, nb=0;
        for (int i=0; i<a.length; i++) {
            dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return (float)(1.0f - dot / (Math.sqrt(na) * Math.sqrt(nb)));
    }
    
    static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i=0; i<a.length; i++) s += Math.abs(a[i]-b[i]);
        return s;
    }
    
    static DistFn getDistFn(String m) {
        if ("cosine".equals(m)) return Main::cosine;
        if ("manhattan".equals(m)) return Main::manhattan;
        return Main::euclidean;
    }

    static class FloatIntPair implements Comparable<FloatIntPair> {
        float dist; int id;
        FloatIntPair(float dist, int id) { this.dist = dist; this.id = id; }
        public int compareTo(FloatIntPair o) {
            int c = Float.compare(this.dist, o.dist);
            if (c != 0) return Integer.compare(this.id, o.id);
            return c;
        }
    }
    
    static class BruteForce {
        List<VectorItem> items = new ArrayList<>();
        public void insert(VectorItem v) { items.add(v); }
        public List<FloatIntPair> knn(float[] q, int k, DistFn dist) {
            List<FloatIntPair> r = new ArrayList<>(items.size());
            for (VectorItem v : items) r.add(new FloatIntPair(dist.compute(q, v.emb), v.id));
            Collections.sort(r);
            if (r.size() > k) r = r.subList(0, k);
            return r;
        }
        public void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }
    
    static class KDNode {
        VectorItem item;
        KDNode left = null;
        KDNode right = null;
        public KDNode(VectorItem v) { item = v; }
    }
    
    static class KDTree {
        KDNode root = null;
        int dims;
        public KDTree(int d) { dims = d; }
        
        KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d+1);
            else n.right = ins(n.right, v, d+1);
            return n;
        }
        
        void knnSearch(KDNode n, float[] q, int k, int d, DistFn dist, PriorityQueue<FloatIntPair> heap) {
            if (n == null) return;
            float dn = dist.compute(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek().dist) {
                heap.add(new FloatIntPair(dn, n.item.id));
                if (heap.size() > k) heap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;
            knnSearch(closer, q, k, d+1, dist, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek().dist)
                knnSearch(farther, q, k, d+1, dist, heap);
        }
        
        public void insert(VectorItem v) { root = ins(root, v, 0); }
        
        public List<FloatIntPair> knn(float[] q, int k, DistFn dist) {
            PriorityQueue<FloatIntPair> heap = new PriorityQueue<>(Collections.reverseOrder());
            knnSearch(root, q, k, 0, dist, heap);
            List<FloatIntPair> r = new ArrayList<>();
            while(!heap.isEmpty()) r.add(heap.poll());
            Collections.sort(r);
            return r;
        }
        
        public void rebuild(List<VectorItem> items) {
            root = null;
            for (VectorItem v : items) insert(v);
        }
    }
    
    static class HNSW {
        static class Node {
            VectorItem item;
            int maxLyr;
            List<List<Integer>> nbrs;
        }
        Map<Integer, Node> G = new HashMap<>();
        int M, M0, ef_build;
        float mL;
        int topLayer = -1;
        int entryPt = -1;
        Random rng = new Random(42);
        
        public HNSW(int m, int efBuild) {
            this.M = m; this.M0 = 2*m; this.ef_build = efBuild;
            this.mL = (float)(1.0 / Math.log(m));
        }
        
        int randLevel() {
            float u = rng.nextFloat();
            if (u == 0) u = 0.0001f;
            return (int)Math.floor(-Math.log(u) * mL);
        }
        
        List<FloatIntPair> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> vis = new HashSet<>();
            PriorityQueue<FloatIntPair> cands = new PriorityQueue<>(); // min-heap
            PriorityQueue<FloatIntPair> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap
            
            float d0 = dist.compute(q, G.get(ep).item.emb);
            vis.add(ep);
            cands.add(new FloatIntPair(d0, ep));
            found.add(new FloatIntPair(d0, ep));
            
            while (!cands.isEmpty()) {
                FloatIntPair curr = cands.poll();
                if (found.size() >= ef && curr.dist > found.peek().dist) break;
                Node cNode = G.get(curr.id);
                if (lyr >= cNode.nbrs.size()) continue;
                for (int nid : cNode.nbrs.get(lyr)) {
                    if (vis.contains(nid) || !G.containsKey(nid)) continue;
                    vis.add(nid);
                    float nd = dist.compute(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().dist) {
                        cands.add(new FloatIntPair(nd, nid));
                        found.add(new FloatIntPair(nd, nid));
                        if (found.size() > ef) found.poll();
                    }
                }
            }
            List<FloatIntPair> res = new ArrayList<>();
            while(!found.isEmpty()) res.add(found.poll());
            Collections.sort(res);
            return res;
        }
        
        List<Integer> selectNbrs(List<FloatIntPair> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i=0; i<Math.min(cands.size(), maxM); i++) r.add(cands.get(i).id);
            return r;
        }
        
        public void insert(VectorItem item, DistFn dist) {
            int id = item.id;
            int lvl = randLevel();
            Node nd = new Node();
            nd.item = item;
            nd.maxLyr = lvl;
            nd.nbrs = new ArrayList<>(lvl + 1);
            for(int i=0; i<=lvl; i++) nd.nbrs.add(new ArrayList<>());
            G.put(id, nd);
            
            if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }
            
            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<FloatIntPair> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }
            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<FloatIntPair> W = searchLayer(item.emb, ep, ef_build, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                G.get(id).nbrs.set(lc, sel);
                
                for (int nid : sel) {
                    if (!G.containsKey(nid)) continue;
                    Node nNode = G.get(nid);
                    while(nNode.nbrs.size() <= lc) nNode.nbrs.add(new ArrayList<>());
                    List<Integer> conn = nNode.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<FloatIntPair> ds = new ArrayList<>();
                        for (int c : conn) if (G.containsKey(c))
                            ds.add(new FloatIntPair(dist.compute(nNode.item.emb, G.get(c).item.emb), c));
                        Collections.sort(ds);
                        conn.clear();
                        for (int i=0; i<maxM && i<ds.size(); i++) conn.add(ds.get(i).id);
                    }
                }
                if (!W.isEmpty()) ep = W.get(0).id;
            }
            if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
        }
        
        public List<FloatIntPair> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<FloatIntPair> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }
            List<FloatIntPair> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            if (W.size() > k) W = W.subList(0, k);
            return W;
        }
        
        public void remove(int id) {
            if (!G.containsKey(id)) return;
            for (Node nd : G.values()) {
                for (List<Integer> layer : nd.nbrs) {
                    layer.remove(Integer.valueOf(id));
                }
            }
            if (entryPt == id) {
                entryPt = -1;
                for (Integer nid : G.keySet()) {
                    if (nid != id) { entryPt = nid; break; }
                }
            }
            G.remove(id);
        }
        
        static class GraphInfo {
            int topLayer, nodeCount;
            int[] nodesPerLayer, edgesPerLayer;
            static class NV { int id; String metadata, category; int maxLyr; }
            static class EV { int src, dst, lyr; }
            List<NV> nodes = new ArrayList<>();
            List<EV> edges = new ArrayList<>();
        }
        
        public GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer = topLayer;
            gi.nodeCount = G.size();
            int maxL = Math.max(topLayer + 1, 1);
            gi.nodesPerLayer = new int[maxL];
            gi.edgesPerLayer = new int[maxL];
            for (Map.Entry<Integer, Node> e : G.entrySet()) {
                int id = e.getKey(); Node nd = e.getValue();
                GraphInfo.NV nv = new GraphInfo.NV();
                nv.id = id; nv.metadata = nd.item.metadata; nv.category = nd.item.category; nv.maxLyr = nd.maxLyr;
                gi.nodes.add(nv);
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    gi.nodesPerLayer[lc]++;
                    if (lc < nd.nbrs.size()) {
                        for (int nid : nd.nbrs.get(lc)) {
                            if (id < nid) {
                                gi.edgesPerLayer[lc]++;
                                GraphInfo.EV ev = new GraphInfo.EV();
                                ev.src = id; ev.dst = nid; ev.lyr = lc;
                                gi.edges.add(ev);
                            }
                        }
                    }
                }
            }
            return gi;
        }
        public int size() { return G.size(); }
    }
    
    static class VectorDB {
        Map<Integer, VectorItem> store = new HashMap<>();
        BruteForce bf = new BruteForce();
        KDTree kdt;
        HNSW hnsw = new HNSW(16, 200);
        int nextId = 1;
        final int dims;
        
        public VectorDB(int d) { this.dims = d; kdt = new KDTree(d); }
        
        public synchronized int insert(String meta, String cat, float[] emb, DistFn dist) {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
            return v.id;
        }
        
        public synchronized boolean remove(int id) {
            if (!store.containsKey(id)) return false;
            store.remove(id); bf.remove(id); hnsw.remove(id);
            List<VectorItem> rem = new ArrayList<>(store.values());
            kdt.rebuild(rem);
            return true;
        }
        
        static class Hit { int id; String meta, cat; float[] emb; float dist; }
        static class SearchOut { List<Hit> hits = new ArrayList<>(); long us; String algo, metric; }
        
        public synchronized SearchOut search(float[] q, int k, String metric, String algo) {
            DistFn dfn = getDistFn(metric);
            long t0 = System.nanoTime();
            List<FloatIntPair> raw;
            if ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
            else if ("kdtree".equals(algo)) raw = kdt.knn(q, k, dfn);
            else raw = hnsw.knn(q, k, 50, dfn);
            
            long us = (System.nanoTime() - t0) / 1000;
            SearchOut out = new SearchOut();
            out.us = us; out.algo = algo; out.metric = metric;
            for (FloatIntPair p : raw) {
                if (store.containsKey(p.id)) {
                    VectorItem v = store.get(p.id);
                    Hit h = new Hit();
                    h.id = v.id; h.meta = v.metadata; h.cat = v.category; h.emb = v.emb; h.dist = p.dist;
                    out.hits.add(h);
                }
            }
            return out;
        }
        
        static class BenchOut { long bfUs, kdUs, hnswUs; int n; }
        
        public synchronized BenchOut benchmark(float[] q, int k, String metric) {
            DistFn dfn = getDistFn(metric);
            BenchOut b = new BenchOut();
            long t0 = System.nanoTime();
            bf.knn(q, k, dfn); b.bfUs = (System.nanoTime() - t0)/1000;
            
            t0 = System.nanoTime();
            kdt.knn(q, k, dfn); b.kdUs = (System.nanoTime() - t0)/1000;
            
            t0 = System.nanoTime();
            hnsw.knn(q, k, 50, dfn); b.hnswUs = (System.nanoTime() - t0)/1000;
            
            b.n = store.size();
            return b;
        }
        
        public synchronized List<VectorItem> all() {
            return new ArrayList<>(store.values());
        }
        
        public synchronized HNSW.GraphInfo hnswInfo() {
            return hnsw.getInfo();
        }
        
        public synchronized int size() { return store.size(); }
    }
    
    // JSON HELPERS
    static String jS(String s) {
        if (s == null) return "\"\"";
        StringBuilder o = new StringBuilder("\"");
        for (int i=0; i<s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"') o.append("\\\"");
            else if (c == '\\') o.append("\\\\");
            else if (c == '\n') o.append("\\n");
            else if (c == '\r') o.append("\\r");
            else if (c == '\t') o.append("\\t");
            else o.append(c);
        }
        return o.append("\"").toString();
    }
    
    static String jVec(float[] v) {
        StringBuilder o = new StringBuilder("[");
        for(int i=0; i<v.length; i++) {
            if(i>0) o.append(",");
            o.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return o.append("]").toString();
    }
    
    static float[] parseVec(String s) {
        if (s == null || s.isEmpty()) return new float[0];
        String[] parts = s.split(",");
        List<Float> lst = new ArrayList<>();
        for(String p : parts) {
            try { lst.add(Float.parseFloat(p.trim())); } catch(Exception e) {}
        }
        float[] res = new float[lst.size()];
        for(int i=0; i<res.length; i++) res[i] = lst.get(i);
        return res;
    }
    
    static String extractStr(String body, String key) {
        String search = "\"" + key + "\"";
        int p = body.indexOf(search);
        if (p == -1) return "";
        p = body.indexOf(":", p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            if (body.charAt(p) == '"') break;
            if (body.charAt(p) == '\\' && p + 1 < body.length()) {
                p++;
                switch (body.charAt(p)) {
                    case '"':  result.append('"');  break;
                    case '\\': result.append('\\'); break;
                    case 'n':  result.append('\n'); break;
                    case 'r':  result.append('\r'); break;
                    case 't':  result.append('\t'); break;
                    default:   result.append(body.charAt(p)); break;
                }
            } else {
                result.append(body.charAt(p));
            }
            p++;
        }
        return result.toString();
    }
    
    static int extractInt(String body, String key, int def) {
        String search = "\"" + key + "\"";
        int p = body.indexOf(search);
        if (p == -1) return def;
        p = body.indexOf(":", p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        int end = p;
        while (end < body.length() && Character.isDigit(body.charAt(end))) end++;
        try { return Integer.parseInt(body.substring(p, end)); } catch(Exception e) { return def; }
    }
    
    static class ParseBodyRes { String meta, cat; float[] emb; }
    static ParseBodyRes parseBody(String b) {
        ParseBodyRes r = new ParseBodyRes();
        r.meta = extractStr(b, "metadata");
        r.cat = extractStr(b, "category");
        int p = b.indexOf("\"embedding\"");
        if (p != -1) {
            p = b.indexOf("[", p);
            if (p != -1) {
                int e = b.indexOf("]", p);
                if (e != -1) {
                    r.emb = parseVec(b.substring(p+1, e));
                }
            }
        }
        if (r.emb == null) r.emb = new float[0];
        return r;
    }
    
    static void cors(HttpExchange ex) {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }
    
    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] wArray = text.trim().split("\\s+");
        List<String> words = new ArrayList<>();
        for(String w : wArray) if (!w.isEmpty()) words.add(w);
        
        if (words.isEmpty()) return new ArrayList<>();
        if (words.size() <= chunkWords) return Collections.singletonList(text);
        
        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i=0; i<words.size(); i+=step) {
            int end = Math.min(i + chunkWords, words.size());
            StringBuilder chunk = new StringBuilder();
            for (int j=i; j<end; j++) {
                if (j>i) chunk.append(" ");
                chunk.append(words.get(j));
            }
            chunks.add(chunk.toString());
            if (end == words.size()) break;
        }
        return chunks;
    }
    
    static class OllamaClient {
        String host;
        int port;
        public String embedModel = "nomic-embed-text";
        public String genModel = "llama3.2";
        HttpClient client;
        
        public OllamaClient(String h, int p) {
            this.host = h; this.port = p;
            this.client = HttpClient.newBuilder().connectTimeout(java.time.Duration.ofSeconds(3)).build();
        }
        
        String esc(String s) {
            StringBuilder o = new StringBuilder();
            for (int i=0; i<s.length(); i++) {
                char c = s.charAt(i);
                if (c == '"') o.append("\\\"");
                else if (c == '\\') o.append("\\\\");
                else if (c == '\n') o.append("\\n");
                else if (c == '\r') o.append("\\r");
                else if (c == '\t') o.append("\\t");
                else o.append(c);
            }
            return o.toString();
        }
        
        public boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder().uri(URI.create("http://" + host + ":" + port + "/api/tags")).GET().build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                return res.statusCode() == 200;
            } catch(Exception e) { return false; }
        }
        
        float[] parseEmbedding(String body) {
            int p = body.indexOf("\"embedding\"");
            if (p == -1) return new float[0];
            p = body.indexOf("[", p);
            if (p == -1) return new float[0];
            int e = p + 1, depth = 1;
            while (e < body.length() && depth > 0) {
                if (body.charAt(e) == '[') depth++;
                else if (body.charAt(e) == ']') depth--;
                e++;
            }
            return parseVec(body.substring(p+1, e-1));
        }
        
        public float[] embed(String text) {
            try {
                String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/embeddings"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .timeout(java.time.Duration.ofSeconds(30))
                    .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return new float[0];
                return parseEmbedding(res.body());
            } catch(Exception e) { return new float[0]; }
        }
        
        public String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + genModel + "\",\"prompt\":\"" + esc(prompt) + "\",\"stream\":false}";
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/generate"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .timeout(java.time.Duration.ofSeconds(180))
                    .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
                return extractStr(res.body(), "response");
            } catch(Exception e) { return "ERROR: Ollama unavailable. Run: ollama serve"; }
        }
    }
    
    static class DocItem {
        int id; String title; String text; float[] emb;
        public DocItem(int id, String title, String text, float[] emb) {
            this.id = id; this.title = title; this.text = text; this.emb = emb;
        }
    }
    
    static class DocumentDB {
        Map<Integer, DocItem> store = new HashMap<>();
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        int nextId = 1;
        int dims = 0;
        
        public synchronized int insert(String title, String text, float[] emb) {
            if (dims == 0) dims = emb.length;
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", emb);
            hnsw.insert(vi, Main::cosine);
            bf.insert(vi);
            return item.id;
        }
        
        static class DocHit { float dist; DocItem doc; DocHit(float dist, DocItem doc) { this.dist = dist; this.doc = doc; } }
        
        public synchronized List<DocHit> search(float[] q, int k, float max_dist) {
            if (store.isEmpty()) return new ArrayList<>();
            List<FloatIntPair> raw = (store.size() < 10) ? bf.knn(q, k, Main::cosine) : hnsw.knn(q, k, 50, Main::cosine);
            List<DocHit> out = new ArrayList<>();
            for (FloatIntPair p : raw) {
                if (store.containsKey(p.id) && p.dist <= max_dist) {
                    out.add(new DocHit(p.dist, store.get(p.id)));
                }
            }
            return out;
        }
        
        public synchronized boolean remove(int id) {
            if (!store.containsKey(id)) return false;
            store.remove(id); hnsw.remove(id); bf.remove(id);
            return true;
        }
        
        public synchronized List<DocItem> all() { return new ArrayList<>(store.values()); }
        public synchronized int size() { return store.size(); }
        public synchronized int getDims() { return dims; }
    }
    
    static void loadDemo(VectorDB db) {
        DistFn dist = Main::cosine;
        db.insert("Linked List: nodes connected by pointers", "cs",
            new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
            new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
            new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
        db.insert("Calculus: derivatives integrals and limits", "math",
            new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
            new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
            new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
            new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
            new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
            new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
    }
    
    static Map<String, String> getQueryParams(String query) {
        Map<String, String> map = new HashMap<>();
        if (query == null) return map;
        for (String param : query.split("&")) {
            String[] kv = param.split("=", 2);
            if (kv.length == 2) {
                try { map.put(kv[0], URLDecoder.decode(kv[1], "UTF-8")); } catch(Exception e) {}
            } else if (kv.length == 1) {
                try { map.put(kv[0], ""); } catch(Exception e) {}
            }
        }
        return map;
    }
    
    static String readBody(HttpExchange ex) throws IOException {
        InputStream in = ex.getRequestBody();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[1024];
        int n;
        while((n = in.read(buf)) != -1) out.write(buf, 0, n);
        return new String(out.toByteArray(), StandardCharsets.UTF_8);
    }
    
    static void sendJson(HttpExchange ex, String json, int code) throws IOException {
        cors(ex);
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(code, bytes.length);
        OutputStream os = ex.getResponseBody();
        os.write(bytes);
        os.close();
    }
    
    public static void main(String[] args) throws Exception {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);
        
        loadDemo(db);
        
        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }
        
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
        
        server.createContext("/", ex -> {
            String path = ex.getRequestURI().getPath();
            
            if (ex.getRequestMethod().equals("OPTIONS")) {
                cors(ex);
                ex.sendResponseHeaders(204, -1);
                return;
            }
            
            try {
                if (path.equals("/search") && ex.getRequestMethod().equals("GET")) {
                    Map<String, String> qp = getQueryParams(ex.getRequestURI().getQuery());
                    float[] q = parseVec(qp.get("v"));
                    if (q.length != DIMS) {
                        sendJson(ex, "{\"error\":\"need " + DIMS + "D vector\"}", 200); return;
                    }
                    int k = 5; try { k = Integer.parseInt(qp.get("k")); } catch(Exception e) {}
                    String metric = qp.getOrDefault("metric", "cosine");
                    if (metric.isEmpty()) metric = "cosine";
                    String algo = qp.getOrDefault("algo", "hnsw");
                    if (algo.isEmpty()) algo = "hnsw";
                    
                    VectorDB.SearchOut out = db.search(q, k, metric, algo);
                    StringBuilder ss = new StringBuilder("{\"results\":[");
                    for (int i=0; i<out.hits.size(); i++) {
                        if (i>0) ss.append(",");
                        VectorDB.Hit h = out.hits.get(i);
                        ss.append("{\"id\":").append(h.id)
                          .append(",\"metadata\":").append(jS(h.meta))
                          .append(",\"category\":").append(jS(h.cat))
                          .append(",\"distance\":").append(String.format(Locale.US, "%.6f", h.dist))
                          .append(",\"embedding\":").append(jVec(h.emb)).append("}");
                    }
                    ss.append("],\"latencyUs\":").append(out.us)
                      .append(",\"algo\":").append(jS(out.algo))
                      .append(",\"metric\":").append(jS(out.metric)).append("}");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.equals("/insert") && ex.getRequestMethod().equals("POST")) {
                    String body = readBody(ex);
                    ParseBodyRes pb = parseBody(body);
                    if (pb.meta == null || pb.meta.isEmpty() || pb.emb.length != DIMS) {
                        sendJson(ex, "{\"error\":\"invalid body\"}", 200); return;
                    }
                    int id = db.insert(pb.meta, pb.cat, pb.emb, Main::cosine);
                    sendJson(ex, "{\"id\":" + id + "}", 200);
                    return;
                }
                
                if (path.startsWith("/delete/") && ex.getRequestMethod().equals("DELETE")) {
                    try {
                        int id = Integer.parseInt(path.substring(8));
                        boolean ok = db.remove(id);
                        sendJson(ex, "{\"ok\":" + (ok?"true":"false") + "}", 200);
                    } catch(Exception e) { sendJson(ex, "{\"ok\":false}", 200); }
                    return;
                }
                
                if (path.equals("/items") && ex.getRequestMethod().equals("GET")) {
                    List<VectorItem> items = db.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i=0; i<items.size(); i++) {
                        if(i>0) ss.append(",");
                        VectorItem v = items.get(i);
                        ss.append("{\"id\":").append(v.id)
                          .append(",\"metadata\":").append(jS(v.metadata))
                          .append(",\"category\":").append(jS(v.category))
                          .append(",\"embedding\":").append(jVec(v.emb)).append("}");
                    }
                    ss.append("]");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.equals("/benchmark") && ex.getRequestMethod().equals("GET")) {
                    Map<String, String> qp = getQueryParams(ex.getRequestURI().getQuery());
                    float[] q = parseVec(qp.get("v"));
                    if (q.length != DIMS) {
                        sendJson(ex, "{\"error\":\"need " + DIMS + "D vector\"}", 200); return;
                    }
                    int k = 5; try { k = Integer.parseInt(qp.get("k")); } catch(Exception e) {}
                    String metric = qp.getOrDefault("metric", "cosine");
                    if (metric.isEmpty()) metric = "cosine";
                    VectorDB.BenchOut b = db.benchmark(q, k, metric);
                    String res = String.format(Locale.US, "{\"bruteforceUs\":%d,\"kdtreeUs\":%d,\"hnswUs\":%d,\"itemCount\":%d}", 
                        b.bfUs, b.kdUs, b.hnswUs, b.n);
                    sendJson(ex, res, 200);
                    return;
                }
                
                if (path.equals("/hnsw-info") && ex.getRequestMethod().equals("GET")) {
                    HNSW.GraphInfo gi = db.hnswInfo();
                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"topLayer\":").append(gi.topLayer).append(",\"nodeCount\":").append(gi.nodeCount)
                      .append(",\"nodesPerLayer\":[");
                    for(int i=0; i<gi.nodesPerLayer.length; i++) {
                        if(i>0) ss.append(","); ss.append(gi.nodesPerLayer[i]);
                    }
                    ss.append("],\"edgesPerLayer\":[");
                    for(int i=0; i<gi.edgesPerLayer.length; i++) {
                        if(i>0) ss.append(","); ss.append(gi.edgesPerLayer[i]);
                    }
                    ss.append("],\"nodes\":[");
                    for(int i=0; i<gi.nodes.size(); i++) {
                        if(i>0) ss.append(",");
                        HNSW.GraphInfo.NV n = gi.nodes.get(i);
                        ss.append("{\"id\":").append(n.id).append(",\"metadata\":").append(jS(n.metadata))
                          .append(",\"category\":").append(jS(n.category)).append(",\"maxLyr\":").append(n.maxLyr).append("}");
                    }
                    ss.append("],\"edges\":[");
                    for(int i=0; i<gi.edges.size(); i++) {
                        if(i>0) ss.append(",");
                        HNSW.GraphInfo.EV e = gi.edges.get(i);
                        ss.append("{\"src\":").append(e.src).append(",\"dst\":").append(e.dst)
                          .append(",\"lyr\":").append(e.lyr).append("}");
                    }
                    ss.append("]}");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.equals("/doc/insert") && ex.getRequestMethod().equals("POST")) {
                    String body = readBody(ex);
                    String title = extractStr(body, "title");
                    String text = extractStr(body, "text");
                    if (title.isEmpty() || text.isEmpty()) {
                        sendJson(ex, "{\"error\":\"need title and text\"}", 200); return;
                    }
                    List<String> chunks = chunkText(text, 250, 30);
                    List<Integer> ids = new ArrayList<>();
                    for (int i=0; i<chunks.size(); i++) {
                        float[] emb = ollama.embed(chunks.get(i));
                        if (emb.length == 0) {
                            sendJson(ex, "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2\"}", 200);
                            return;
                        }
                        String chunkTitle = chunks.size() > 1 ? title + " [" + (i+1) + "/" + chunks.size() + "]" : title;
                        ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
                    }
                    StringBuilder ss = new StringBuilder("{\"ids\":[");
                    for(int i=0; i<ids.size(); i++) { if(i>0) ss.append(","); ss.append(ids.get(i)); }
                    ss.append("],\"chunks\":").append(chunks.size()).append(",\"dims\":").append(docDB.getDims()).append("}");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.startsWith("/doc/delete/") && ex.getRequestMethod().equals("DELETE")) {
                    try {
                        int id = Integer.parseInt(path.substring(12));
                        boolean ok = docDB.remove(id);
                        sendJson(ex, "{\"ok\":" + (ok?"true":"false") + "}", 200);
                    } catch(Exception e) { sendJson(ex, "{\"ok\":false}", 200); }
                    return;
                }
                
                if (path.equals("/doc/list") && ex.getRequestMethod().equals("GET")) {
                    List<DocItem> docs = docDB.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i=0; i<docs.size(); i++) {
                        if(i>0) ss.append(",");
                        DocItem d = docs.get(i);
                        String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
                        int words = d.text.split("\\s+").length;
                        ss.append("{\"id\":").append(d.id)
                          .append(",\"title\":").append(jS(d.title))
                          .append(",\"preview\":").append(jS(preview))
                          .append(",\"words\":").append(words).append("}");
                    }
                    ss.append("]");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.equals("/doc/search") && ex.getRequestMethod().equals("POST")) {
                    String body = readBody(ex);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) { sendJson(ex, "{\"error\":\"need question\"}", 200); return; }
                    
                    float[] qEmb = ollama.embed(question);
                    if (qEmb.length == 0) { sendJson(ex, "{\"error\":\"Ollama unavailable\"}", 200); return; }
                    
                    List<DocumentDB.DocHit> hits = docDB.search(qEmb, k, 0.7f); // max_dist from c++
                    StringBuilder ss = new StringBuilder("{\"contexts\":[");
                    for(int i=0; i<hits.size(); i++) {
                        if(i>0) ss.append(",");
                        ss.append("{\"id\":").append(hits.get(i).doc.id)
                          .append(",\"title\":").append(jS(hits.get(i).doc.title))
                          .append(",\"distance\":").append(String.format(Locale.US, "%.4f", hits.get(i).dist)).append("}");
                    }
                    ss.append("]}");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.equals("/doc/ask") && ex.getRequestMethod().equals("POST")) {
                    String body = readBody(ex);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) { sendJson(ex, "{\"error\":\"need question\"}", 200); return; }
                    
                    float[] qEmb = ollama.embed(question);
                    if (qEmb.length == 0) { sendJson(ex, "{\"error\":\"Ollama unavailable\"}", 200); return; }
                    
                    List<DocumentDB.DocHit> hits = docDB.search(qEmb, k, 0.7f);
                    StringBuilder ctx = new StringBuilder();
                    for(int i=0; i<hits.size(); i++) {
                        ctx.append("[").append(i+1).append("] ").append(hits.get(i).doc.title).append(":\n")
                           .append(hits.get(i).doc.text).append("\n\n");
                    }
                    String prompt = "You are a helpful assistant. Answer the user's question directly. "
                        + "Use the provided context if it contains relevant information. "
                        + "If it doesn't, just use your own general knowledge. "
                        + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                        + "Just answer the question naturally.\n\n"
                        + "Context:\n" + ctx.toString()
                        + "Question: " + question + "\n\n"
                        + "Answer:";
                    
                    String answer = ollama.generate(prompt);
                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"answer\":").append(jS(answer))
                      .append(",\"model\":").append(jS(ollama.genModel))
                      .append(",\"contexts\":[");
                    for(int i=0; i<hits.size(); i++) {
                        if(i>0) ss.append(",");
                        ss.append("{\"id\":").append(hits.get(i).doc.id)
                          .append(",\"title\":").append(jS(hits.get(i).doc.title))
                          .append(",\"text\":").append(jS(hits.get(i).doc.text))
                          .append(",\"distance\":").append(String.format(Locale.US, "%.4f", hits.get(i).dist)).append("}");
                    }
                    ss.append("],\"docCount\":").append(docDB.size()).append("}");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.equals("/status") && ex.getRequestMethod().equals("GET")) {
                    boolean up = ollama.isAvailable();
                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"ollamaAvailable\":").append(up ? "true" : "false")
                      .append(",\"embedModel\":").append(jS(ollama.embedModel))
                      .append(",\"genModel\":").append(jS(ollama.genModel))
                      .append(",\"docCount\":").append(docDB.size())
                      .append(",\"docDims\":").append(docDB.getDims())
                      .append(",\"demoDims\":").append(DIMS)
                      .append(",\"demoCount\":").append(db.size()).append("}");
                    sendJson(ex, ss.toString(), 200);
                    return;
                }
                
                if (path.equals("/stats") && ex.getRequestMethod().equals("GET")) {
                    String res = "{\"count\":" + db.size() + ",\"dims\":" + DIMS + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
                    sendJson(ex, res, 200);
                    return;
                }
                
                if (path.equals("/") && ex.getRequestMethod().equals("GET")) {
                    File file = new File("index.html");
                    if (!file.exists()) {
                        ex.sendResponseHeaders(404, -1);
                        return;
                    }
                    byte[] bytes = Files.readAllBytes(file.toPath());
                    ex.getResponseHeaders().add("Content-Type", "text/html");
                    ex.sendResponseHeaders(200, bytes.length);
                    OutputStream os = ex.getResponseBody();
                    os.write(bytes);
                    os.close();
                    return;
                }
                
                ex.sendResponseHeaders(404, -1);
                
            } catch(Exception e) {
                e.printStackTrace();
                ex.sendResponseHeaders(500, -1);
            }
        });
        
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
    }
}
