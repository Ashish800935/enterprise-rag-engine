# Enterprise Hybrid-RAG Engine with PostgreSQL (pgvector) & FastAPI

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20+%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready, containerized **Hybrid-RAG (Retrieval-Augmented Generation)** knowledge system built using **FastAPI**, **PostgreSQL (`pgvector`)**, and **Streamlit**. Engineered with an emphasis on low-latency vector search, anti-hallucination prompt constraints, and full containerized orchestration.

---

## 🏗️ Architecture

```mermaid
graph TD
    User([Client / Streamlit UI]) -->|Upload PDF/TXT/MD| API[FastAPI Backend Engine]
    User -->|Submit Natural Language Query| API
    
    subgraph Ingestion Pipeline
        API -->|1. Overlapping Split| Chunker[Semantic Chunker (500 chars, 50 overlap)]
        Chunker -->|2. Batch Embedding| Model[SentenceTransformers: all-MiniLM-L6-v2]
        Model -->|3. Store Vectors| PG[(PostgreSQL + pgvector HNSW Index)]
    end
    
    subgraph Retrieval & Synthesis
        API -->|4. Vector Cosine Distance + Keyword Match| PG
        PG -->|5. Top-K Ranked Chunks| Reranker[Context Filter & Scoring]
        Reranker -->|6. Grounded Prompt Injection| LLM[LLM Synthesizer]
        LLM -->|7. Verified Answer + Citations| API
    end
    
    API -->|8. Answer with Latency Metrics & Sources| User
```

---

## 🌟 Key Engineering Features

1. **Native PostgreSQL Vector Storage (`pgvector`)**:
   - Eliminates the need for external vector databases (Pinecone/Milvus).
   - Relational document metadata and 384-dimensional dense vectors live in the same ACID-compliant database.
   - Utilizes **HNSW (Hierarchical Navigable Small World)** indexing for sub-millisecond approximate nearest neighbor search.

2. **Hybrid Retrieval (Dense + Sparse Boost)**:
   - Combines dense semantic vector search (`<=>` cosine distance operator) with sparse lexical keyword boosting.
   - Prevents vector search blind spots on exact identifiers, codes, and acronyms.

3. **Grounded Synthesis & Source Attribution**:
   - Strict system prompts forcing responses to be grounded in retrieved chunks.
   - Explicit citation tracking: every statement maps back to its specific document filename, chunk index, and similarity score.

4. **Production Containerization**:
   - Fully automated multi-container orchestration with `docker-compose.yml` (Postgres, FastAPI, Streamlit).

---

## 📁 Project Structure

```text
enterprise-rag-engine/
├── docker-compose.yml           # Multi-container orchestration
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules (keeps secrets & venvs safe)
├── README.md                    # Project overview & quickstart
├── ARCHITECTURE.md              # Technical architecture & design decisions
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py              # FastAPI app with lifespan startup
│       ├── config.py            # Pydantic settings
│       ├── db/
│       │   ├── session.py       # SQLAlchemy engine & pgvector initialization
│       │   └── models.py        # Document & DocumentChunk with HNSW Vector index
│       ├── services/
│       │   ├── chunking.py      # Recursive sliding-window chunker
│       │   ├── embedding.py     # Singleton SentenceTransformers embedder
│       │   ├── retrieval.py     # Cosine distance vector & hybrid search
│       │   └── generator.py     # Grounded response synthesis
│       └── api/
│           ├── schemas.py       # Pydantic request/response models
│           └── routes.py        # Ingestion, Query, and Health endpoints
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                   # Streamlit UI with vector inspector
```

---

## 🚀 Quickstart (One-Click Docker Setup)

### 1. Clone & Configure
```bash
git clone https://github.com/Ashish800935/enterprise-rag-engine.git
cd enterprise-rag-engine
cp .env.example .env
```

### 2. Launch with Docker Compose
```bash
docker compose up --build
```

The services will be live at:
* **Interactive Streamlit UI:** [http://localhost:8501](http://localhost:8501)
* **FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **PostgreSQL Database:** `localhost:5433` (`rag_db`)

---

## 🛠️ Local Development (Without Docker)

If you wish to run services directly on your host machine:

### 1. Start PostgreSQL with pgvector
```bash
docker run -d --name rag_postgres -p 5433:5432 -e POSTGRES_DB=rag_db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres_password pgvector/pgvector:pg16
```

### 2. Run FastAPI Backend
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

### 3. Run Streamlit Frontend
```bash
cd ../frontend
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📡 API Reference

### `POST /api/v1/documents/upload`
Uploads and indexes a document (`.txt`, `.md`, `.pdf`).
* **Request:** Multipart Form-Data (`file`).
* **Response:** Document metadata with `total_chunks` generated.

### `POST /api/v1/query`
Executes hybrid vector retrieval and returns grounded answer.
* **Request Body:**
  ```json
  {
    "query": "What are the main findings in Section 3?",
    "top_k": 4,
    "use_hybrid": true
  }
  ```
* **Response:**
  ```json
  {
    "query": "What are the main findings in Section 3?",
    "answer": "...",
    "sources": [
      {
        "chunk_id": 12,
        "document_id": 1,
        "filename": "annual_report.pdf",
        "chunk_index": 3,
        "similarity_score": 0.884,
        "content": "..."
      }
    ],
    "retrieval_latency_ms": 14.2,
    "total_latency_ms": 820.5
  }
  ```

---

## 📖 Deep Dive & System Design
For an in-depth breakdown of architectural trade-offs, vector distance mathematics, HNSW indexing parameters, and scaling strategies, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 👨‍💻 Author
**Ashish**
* GitHub: [@Ashish800935](https://github.com/Ashish800935)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
