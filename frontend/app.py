import streamlit as st
import requests
import os

# Configuration: Supports both local env vars and Streamlit Cloud secrets
BACKEND_URL = os.getenv("BACKEND_API_URL")
if not BACKEND_URL:
    try:
        BACKEND_URL = st.secrets.get("BACKEND_API_URL", "http://localhost:8000")
    except Exception:
        BACKEND_URL = "http://localhost:8000"

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
    .badge-high {
        background-color: #198754;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .badge-mid {
        background-color: #fd7e14;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .badge-low {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚙️ Knowledge Base")
    
    # 1. System Health Status
    try:
        health_res = requests.get(f"{API_BASE}/health", timeout=15)
        if health_res.status_code == 200:
            health_data = health_res.json()
            if "connected" in health_data.get("database", ""):
                st.success(f"🟢 **System Online**\n\nDB: `{health_data['database']}`")
            else:
                st.warning(f"🟡 **DB Reconnecting...**\n\n`{health_data.get('database')}`")
        elif health_res.status_code in (502, 503, 504):
            st.info("🟡 **Server Waking Up...**\n\nFree cloud instances sleep after 15m of inactivity. Takes ~25s to wake up.")
            if st.button("🔄 Refresh Status", use_container_width=True):
                st.rerun()
        else:
            st.error(f"🔴 Backend Status: {health_res.status_code}")
    except Exception:
        st.info("🟡 **Server Waking Up...**\n\nFree cloud instances sleep after 15m of inactivity. Takes ~25s to wake up.")
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    st.divider()

    # 2. Document Upload Form
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"],
        help="Documents are split using LangChain's RecursiveCharacterTextSplitter, embedded into 384-d vectors, and indexed in PostgreSQL (pgvector)."
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 15:
            st.warning(f"⚠️ Large file ({file_size_mb:.1f} MB). Free cloud tier may timeout on files > 15 MB. For best performance, use files under 10 MB.")

        if st.button("🚀 Ingest & Index", use_container_width=True):
            with st.spinner("LangChain Text Splitting & Dense Embedding..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{API_BASE}/documents/upload", files=files, timeout=180)
                    if res.status_code == 201:
                        data = res.json()
                        st.success(f"✅ Indexed **{data['filename']}** ({data['total_chunks']} chunks created)!")
                        st.rerun()
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    st.divider()

    # 3. Document Explorer
    st.subheader("📚 Ingested Documents")
    try:
        docs_res = requests.get(f"{API_BASE}/documents", timeout=10)
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
st.caption("PostgreSQL `pgvector` + FastAPI + LangChain LCEL & ReAct Tools")

# Mode Switcher
mode = st.radio(
    "Select Query Pipeline Mode:",
    [
        "🌟 LangChain Structured RAG (LCEL)",
        "🤖 LangChain Agentic Mode (Tools)",
        "⚡ Standard Direct RAG"
    ],
    horizontal=True
)

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Ask a question grounded in your uploaded documents:",
        value=st.session_state.search_query,
        placeholder="e.g., What are the key findings in Section 3? Or ask agent: 'How many documents exist?'"
    )
with col2:
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)
    use_hybrid = st.checkbox("Hybrid Retrieval (Keyword Boost)", value=True)

if st.button("🔎 Execute Query", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        # ================= MODE 1: LANGCHAIN STRUCTURED RAG =================
        if "Structured RAG" in mode:
            with st.spinner("Executing LangChain LCEL Chain & Pydantic Structured Output..."):
                try:
                    payload = {"query": query, "top_k": top_k, "use_hybrid": use_hybrid}
                    res = requests.post(f"{API_BASE}/query/structured", json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        conf = data["confidence_score"]
                        
                        # Metrics Row
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Retrieval Latency", f"{data['retrieval_latency_ms']} ms")
                        m2.metric("Total Latency", f"{data['total_latency_ms']} ms")
                        
                        conf_pct = int(conf * 100)
                        if conf >= 0.7:
                            m3.metric("Grounding Confidence", f"{conf_pct}%", delta="High Grounding", delta_color="normal")
                        elif conf >= 0.4:
                            m3.metric("Grounding Confidence", f"{conf_pct}%", delta="Moderate Grounding", delta_color="off")
                        else:
                            m3.metric("Grounding Confidence", f"{conf_pct}%", delta="Low Grounding", delta_color="inverse")

                        st.divider()

                        # Answer Section
                        st.subheader("💡 Grounded Answer (LCEL Synthesized)")
                        st.markdown(data["answer"])

                        st.divider()

                        # Citations
                        st.subheader("📑 Structured Citations")
                        if data.get("citations"):
                            for idx, cit in enumerate(data["citations"], 1):
                                with st.expander(f"Citation #{idx}: `{cit['filename']}` (Chunk #{cit['chunk_index']})"):
                                    st.info(f"**Verbatim Quote:** \"{cit['exact_quote']}\"")
                        else:
                            st.caption("No specific citations generated.")

                        # Suggested Follow-up Questions
                        if data.get("suggested_followups"):
                            st.divider()
                            st.subheader("💡 Suggested Follow-up Questions")
                            for follow_up in data["suggested_followups"]:
                                if st.button(f"👉 {follow_up}", key=f"fup_{follow_up}"):
                                    st.session_state.search_query = follow_up
                                    st.rerun()
                    else:
                        st.error(f"Backend returned error: {res.text}")
                except Exception as e:
                    st.error(f"Could not reach backend API: {e}")

        # ================= MODE 2: LANGCHAIN AGENTIC MODE =================
        elif "Agentic Mode" in mode:
            with st.spinner("LangChain ReAct Agent reasoning and executing tools..."):
                try:
                    payload = {"query": query, "top_k": top_k, "use_hybrid": use_hybrid}
                    res = requests.post(f"{API_BASE}/query/agent", json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        
                        # Metrics Row
                        m1, m2 = st.columns(2)
                        m1.metric("Execution Latency", f"{data['latency_ms']} ms")
                        m2.metric("Agent Steps Completed", data["total_steps"])

                        st.divider()

                        # Tools Invoked
                        st.subheader("🛠️ Tools Selected by Agent")
                        if data.get("tools_used"):
                            st.write(" ".join([f"`{t}`" for t in data["tools_used"]]))
                        else:
                            st.write("*Direct reasoning (no external tools required)*")

                        # Reasoning Trail
                        if data.get("steps"):
                            with st.expander("🔍 View Agent Thought Process & Tool Execution Trail", expanded=True):
                                for s in data["steps"]:
                                    st.markdown(f"**Step {s['step_number']} - Thought:** {s['thought']}")
                                    st.markdown(f"- **Tool Called:** `{s['tool_name']}`")
                                    st.markdown(f"- **Input:** `{s['tool_input']}`")
                                    st.info(f"**Tool Output:** {s['tool_output']}")
                                    st.markdown("---")

                        # Final Answer
                        st.subheader("🤖 Final Agent Answer")
                        st.markdown(data["final_answer"])
                    else:
                        st.error(f"Backend returned error: {res.text}")
                except Exception as e:
                    st.error(f"Could not reach backend API: {e}")

        # ================= MODE 3: STANDARD DIRECT RAG =================
        else:
            with st.spinner("Executing vector similarity search & LLM synthesis..."):
                try:
                    payload = {"query": query, "top_k": top_k, "use_hybrid": use_hybrid}
                    res = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
                    if res.status_code == 200:
                        data = res.json()
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Retrieval Latency", f"{data['retrieval_latency_ms']} ms")
                        m2.metric("Total Latency", f"{data['total_latency_ms']} ms")
                        m3.metric("Chunks Retrieved", len(data['sources']))

                        st.divider()
                        st.subheader("💡 Grounded Answer")
                        st.markdown(data["answer"])

                        st.divider()
                        st.subheader("🔍 Retrieved Context Chunks & Similarity Inspector")
                        for idx, src in enumerate(data["sources"], start=1):
                            score_pct = round(src["similarity_score"] * 100, 1)
                            with st.expander(f"Chunk #{src['chunk_index']} from `{src['filename']}` — Match Score: {score_pct}%"):
                                st.markdown(f"**Document ID:** `{src['document_id']}` | **Chunk ID:** `{src['chunk_id']}`")
                                st.info(src["content"])
                    else:
                        st.error(f"Backend returned error: {res.text}")
                except Exception as e:
                    st.error(f"Could not reach backend API: {e}")
