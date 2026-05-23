# AI-Powered Vector Database & RAG Engine

![Model Image](model%20image.png)

A high-performance Vector Database and Retrieval-Augmented Generation (RAG) system built completely in Java from scratch.

This project was created to deeply understand how modern AI retrieval systems work internally instead of relying on external vector databases or frameworks. The system supports semantic vector search, document retrieval, ANN indexing, benchmarking, and LLM-powered question answering through Ollama.

The goal was simple:

> Build the core infrastructure behind modern AI search systems like ChatGPT memory, semantic search engines, and RAG pipelines using only Java and fundamental data structures.

---

# Features

## Vector Similarity Search

Supports multiple search approaches:

* HNSW (Hierarchical Navigable Small World)
* KD-Tree
* Brute Force Search

## Distance Metrics

Implemented multiple similarity functions:

* Cosine Similarity
* Euclidean Distance
* Manhattan Distance

## AI-Powered RAG Pipeline

* Document ingestion
* Automatic chunking
* Embedding generation using Ollama
* Semantic retrieval
* Context-aware AI responses

## REST API Server

Built a lightweight HTTP server using Java's native `HttpServer`.

Supports:

* Insert vectors
* Delete vectors
* Search vectors
* Benchmark algorithms
* Upload documents
* Ask questions to AI
* Retrieve statistics

## Benchmarking Engine

Compare performance of:

* Brute Force
* KD-Tree
* HNSW

Measure:

* Search latency
* Retrieval quality
* Scalability behavior

## HNSW Graph Visualization Support

Exposes graph structure data:

* Nodes
* Layers
* Edges
* Connectivity

Useful for understanding ANN internals.

---

# Why I Built This

Most developers use vector databases as black boxes.

I wanted to understand:

* How semantic search actually works
* Why HNSW is fast
* How embeddings are stored and retrieved
* How RAG systems work internally
* How modern AI assistants retrieve memory/context

Instead of using Pinecone, Weaviate, ChromaDB, or FAISS directly, I decided to implement the core concepts manually in Java.

This project became a deep dive into:

* Data structures
* Approximate nearest neighbor search
* AI retrieval systems
* Distributed-search concepts
* Embedding pipelines
* LLM integration

---

# System Architecture

```text
User Query
    ↓
Embedding Generation (Ollama)
    ↓
Vector Search Engine
    ├── HNSW
    ├── KD-Tree
    └── Brute Force
    ↓
Top-K Relevant Chunks
    ↓
Context Construction
    ↓
LLM Response Generation
    ↓
Final AI Answer
```

---

# Core Components

# 1. Vector Database Engine

The VectorDB acts as the central storage and retrieval system.

Each vector item contains:

* Unique ID
* Metadata
* Category
* Embedding vector

Supports:

* Insertions
* Deletions
* Similarity search
* Benchmarking

---

# 2. HNSW Implementation

One of the most important parts of the project.

HNSW is an Approximate Nearest Neighbor algorithm that creates multi-layer graph structures for extremely fast vector retrieval.

### What I implemented

* Random layer generation
* Multi-layer graph traversal
* Neighbor selection
* Dynamic insertion
* Search-layer exploration
* Graph pruning
* Entry-point optimization

### Why it matters

HNSW drastically reduces search time compared to brute force search while maintaining strong retrieval accuracy.

This is the same family of ANN algorithms used in production-grade AI retrieval systems.

---

# 3. KD-Tree Implementation

Implemented recursive KD-Tree indexing for structured nearest-neighbor search.

### Features

* Recursive node insertion
* Axis-based partitioning
* Backtracking nearest-neighbor search
* Tree rebuilding after deletions

### Limitation

KD-Trees perform well in lower dimensions but degrade in high-dimensional embedding spaces.

This became one of the interesting observations during benchmarking.

---

# 4. Brute Force Search

The baseline retrieval algorithm.

Calculates distance against every vector directly.

### Why I included it

* Accuracy comparison
* Benchmarking baseline
* Validation of ANN search results

Although slower, it guarantees exact nearest-neighbor retrieval.

---

# 5. Document RAG Pipeline

Implemented a complete Retrieval-Augmented Generation pipeline.

### Flow

1. Upload document
2. Split into chunks
3. Generate embeddings
4. Store embeddings in vector database
5. Retrieve relevant chunks during query
6. Pass context to LLM
7. Generate final answer

### Chunking Strategy

Used overlapping chunk windows to preserve semantic continuity between sections.

This significantly improved retrieval quality.

---

# 6. Ollama Integration

Integrated local LLM inference using Ollama.

### Models Used

* `nomic-embed-text`
* `llama3.2`

### Capabilities

* Embedding generation
* AI answer generation
* Offline local inference
* No external API dependency

This allowed the entire AI pipeline to run locally.

---

# API Endpoints

## Vector APIs

| Endpoint       | Method | Description               |
| -------------- | ------ | ------------------------- |
| `/insert`      | POST   | Insert vector             |
| `/search`      | GET    | Similarity search         |
| `/delete/{id}` | DELETE | Remove vector             |
| `/items`       | GET    | Retrieve all vectors      |
| `/benchmark`   | GET    | Compare search algorithms |
| `/stats`       | GET    | Database statistics       |

---

## Document APIs

| Endpoint           | Method | Description           |
| ------------------ | ------ | --------------------- |
| `/doc/insert`      | POST   | Upload document       |
| `/doc/search`      | POST   | Semantic retrieval    |
| `/doc/ask`         | POST   | Ask AI questions      |
| `/doc/delete/{id}` | DELETE | Delete document       |
| `/doc/list`        | GET    | List stored documents |

---

# Performance Observations

During testing:

### Brute Force

* Most accurate
* Slowest at scale

### KD-Tree

* Fast on lower dimensions
* Performance degraded with larger embedding spaces

### HNSW

* Best scalability
* Extremely fast retrieval
* Excellent balance between speed and accuracy

This project helped visualize why modern vector databases heavily rely on ANN indexing.

---

# Challenges Faced

## 1. Building HNSW From Scratch

The hardest part of the project.

Challenges included:

* Multi-layer graph management
* Efficient neighbor pruning
* Maintaining graph quality
* Balancing speed vs accuracy
* Handling dynamic insertion

Debugging graph traversal logic took significant time.

---

## 2. Manual JSON Parsing

I intentionally avoided external frameworks to understand low-level backend handling.

This meant:

* Parsing requests manually
* Escaping JSON safely
* Handling malformed requests
* Managing serialization/deserialization manually

It was difficult but taught a lot about backend internals.

---

## 3. Concurrency & Synchronization

Since the server handles concurrent requests, thread safety became important.

Used:

* synchronized methods
* thread-safe operations
* executor-based request handling

Preventing race conditions during insertions/deletions was challenging.

---

## 4. Embedding Dimension Handling

Different embedding models can produce different dimensions.

I had to:

* validate vector sizes
* dynamically manage embedding dimensions
* prevent inconsistent insertions

---

## 5. Balancing Retrieval Accuracy

Tuning:

* HNSW parameters
* ef values
* graph connections
* chunk sizes

had major impact on retrieval quality and performance.

---

# Tech Stack

## Languages

* Java

## AI & ML

* Ollama
* Llama 3.2
* Nomic Embed Text

## Algorithms

* HNSW
* KD-Tree
* ANN Search
* Brute Force KNN

## Backend

* Java HttpServer
* REST APIs
* Concurrent Processing

---

# What I Learned

This project taught me:

* How vector databases work internally
* ANN search algorithms
* Semantic retrieval systems
* RAG architecture
* Embedding pipelines
* Backend system design
* Performance optimization
* AI infrastructure fundamentals

More importantly, it helped bridge the gap between:

> "Using AI tools" and "Building AI systems."

---

# Future Improvements

Planned upgrades:

* Persistent disk storage
* Quantization support
* Hybrid search
* Distributed indexing
* GPU acceleration
* Better graph visualization
* Authentication system
* Streaming responses
* WebSocket support

---

# Running the Project

## Requirements

* Java 17+
* Ollama installed locally

Install models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

Run the project:

```bash
javac Main.java
java Main
```

Server starts at:

```text
http://localhost:8080
```

---

# Example Query

```bash
GET /search?v=0.1,0.2,0.3...&k=5&algo=hnsw&metric=cosine
```

---

# Final Thoughts

This project started as an experiment to understand vector search systems.

It eventually became a complete AI retrieval engine with:

* ANN indexing
* Semantic search
* Document retrieval
* Local LLM integration
* RAG pipeline support

Building everything from scratch gave me a much deeper understanding of how modern AI infrastructure works behind the scenes.
