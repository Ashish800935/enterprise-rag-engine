import streamlit as st
import requests
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

st.set_page_config(
    page_title="Enterprise Hybrid-RAG Engine",
    page_icon="🧠",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e24;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #333;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ Knowledge Base")
    
    # 1. System Health Status
    try:
        health_res = requests.get(f"{API_BASE}/health", timeout=3)
        if health_res.status_code == 200:
            health_data = health_res.json()
            st.success(f"🟢 **System Online**\n\nDB: `{health_data['database']}`")
        else:
            st.error("🔴 Backend Error")
    except Exception:
        st.warning("🟡 Backend Disconnected (Start FastAPI on port 8000)")

    st.divider()

    # 2. Document Upload Form
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        help="Documents are chunked, embedded into 384-d vectors, and indexed in PostgreSQL (pgvector)."
    )

    if uploaded_file is not None:
        if st.button("🚀 Ingest & Index", use_container_width=True):
            with st.spinner("Chunking text & generating dense embeddings..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{API_BASE}/documents/upload", files=files)
                    if res.status_code == 201:
                        data = res.json()
                        st.success(f"✅ Indexed **{data['filename']}** ({data['total_chunks']} chunks created)!")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    st.divider()

    # 3. Document Explorer
    st.subheader("📚 Ingested Documents")
    try:
        docs_res = requests.get(f"{API_BASE}/documents", timeout=3)
        if docs_res.status_code == 200:
            docs = docs_res.json()
            if not docs:
                st.info("No documents uploaded yet.")
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 **{doc['filename']}** ({doc['total_chunks']} chunks)")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document & vector chunks"):
                        requests.delete(f"{API_BASE}/documents/{doc['id']}")
                        st.rerun()
    except Exception:
        pass


# ----------------- MAIN QUERY CANVAS -----------------
st.title("🧠 Enterprise Hybrid-RAG Engine")
st.caption("Containerized Production Retrieval-Augmented Generation with PostgreSQL `pgvector`, FastAPI & Streamlit")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Ask a question grounded in your uploaded documents:",
        placeholder="e.g., What are the key findings in the annual report?"
    )
with col2:
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)
    use_hybrid = st.checkbox("Hybrid Retrieval (Keyword Boost)", value=True)

if st.button("🔎 Run Grounded Query", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Executing vector similarity search & LLM synthesis..."):
            try:
                payload = {
                    "query": query,
                    "top_k": top_k,
                    "use_hybrid": use_hybrid
                }
                res = requests.post(f"{API_BASE}/query", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    
                    # Latency Telemetry
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Retrieval Latency", f"{data['retrieval_latency_ms']} ms")
                    m2.metric("Total Latency", f"{data['total_latency_ms']} ms")
                    m3.metric("Chunks Retrieved", len(data['sources']))

                    st.divider()

                    # Answer Section
                    st.subheader("💡 Grounded Answer")
                    st.markdown(data["answer"])

                    st.divider()

                    # Citations and Vector Inspector
                    st.subheader("🔍 Retrieved Context Chunks & Similarity Inspector")
                    st.caption("Inspect the exact source chunks retrieved from PostgreSQL pgvector with their cosine similarity scores:")
                    
                    for idx, src in enumerate(data["sources"], start=1):
                        score_pct = round(src["similarity_score"] * 100, 1)
                        with st.expander(f"Chunk #{src['chunk_index']} from `{src['filename']}` — Match Score: {score_pct}%"):
                            st.markdown(f"**Document ID:** `{src['document_id']}` | **Chunk ID:** `{src['chunk_id']}`")
                            st.info(src["content"])
                else:
                    st.error(f"Backend returned error: {res.text}")
            except Exception as e:
                st.error(f"Could not reach backend API: {e}")
