from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.orm import Session
from src.config import settings
from src.db.models import Document, DocumentChunk
from src.services.retrieval import RetrievalService
from duckduckgo_search import DDGS
import logging
import json

logger = logging.getLogger("rag.agent")

# Models for Agent Telemetry
class AgentExecutionStep(BaseModel):
    step_number: int
    thought: str
    tool_name: str
    tool_input: str
    tool_output: str

class AgentResponse(BaseModel):
    query: str
    final_answer: str
    tools_used: List[str]
    steps: List[AgentExecutionStep]
    total_steps: int

# 1. Tool Definitions
def create_knowledge_base_search_tool(db: Session):
    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Searches the enterprise knowledge base (PostgreSQL with pgvector) for semantically relevant chunks.
        Use this tool whenever the user asks questions about uploaded documents, policies, reports, or indexed data.
        """
        chunks = RetrievalService.hybrid_search(db, query, top_k=4)
        if not chunks:
            return "No matching documents or chunks found in the enterprise knowledge base."
        
        output_parts = []
        for c in chunks:
            output_parts.append(
                f"[Document: {c['filename']} (Chunk #{c['chunk_index']}) | Relevance: {c['similarity_score']}]\n{c['content']}"
            )
        return "\n\n".join(output_parts)
    
    return search_knowledge_base

def create_document_stats_tool(db: Session):
    @tool
    def get_document_stats(query: str = "") -> str:
        """
        Retrieves high-level metadata and statistics about the knowledge base.
        Returns the total number of ingested documents, total vector chunks, and list of file names.
        Use this when the user asks 'how many documents exist?', 'list uploaded files', or 'knowledge base status'.
        """
        docs = db.query(Document).all()
        total_chunks = db.query(DocumentChunk).count()
        if not docs:
            return "The knowledge base is currently empty. No documents have been uploaded."
        
        filenames = [f"- {d.filename} ({d.total_chunks} chunks, {round(d.file_size_bytes/1024, 1)} KB)" for d in docs]
        return (
            f"Knowledge Base Statistics:\n"
            f"- Total Ingested Documents: {len(docs)}\n"
            f"- Total Vector Chunks in PostgreSQL: {total_chunks}\n"
            f"Uploaded Files:\n" + "\n".join(filenames)
        )
    
    return get_document_stats

@tool
def web_search_fallback(query: str) -> str:
    """
    Performs a live web search using DuckDuckGo.
    Use this tool ONLY when:
    1. The knowledge base search returned no results, OR
    2. The user explicitly requests real-time, external, or internet-wide information.
    """
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return f"No external web results found for '{query}'."
        
        snippets = []
        for r in results:
            snippets.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
        return "\n\n".join(snippets)
    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}")
        return f"Web search could not be completed: {str(e)}"

# 2. Agent Execution Engine
AGENT_SYSTEM_PROMPT = """You are an Enterprise AI Agent equipped with specialized tools:
1. `search_knowledge_base`: Use this for any factual questions about enterprise documents.
2. `get_document_stats`: Use this when asked about system counts, uploaded documents, or knowledge base metrics.
3. `web_search_fallback`: Use this if the knowledge base does not contain the answer and external knowledge is required.

OPERATIONAL INSTRUCTIONS:
- First, evaluate which tool is most appropriate for the user's intent.
- Always provide transparent, helpful, and concise answers based on tool observations.
- Do not fabricate information. Cite tool results directly.
"""

def run_langchain_agent(db: Session, query: str) -> AgentResponse:
    """
    Executes a LangChain Tool-Calling Agent.
    Routes between `search_knowledge_base`, `get_document_stats`, and `web_search_fallback`.
    Records execution steps for telemetry and UI visualization.
    """
    tool_kb = create_knowledge_base_search_tool(db)
    tool_stats = create_document_stats_tool(db)
    tool_web = web_search_fallback
    
    tools = [tool_kb, tool_stats, tool_web]
    tool_map = {t.name: t for t in tools}

    steps: List[AgentExecutionStep] = []
    tools_used: List[str] = []

    # 1. If Gemini API Key is available, run full LLM Tool-Calling loop
    if settings.GEMINI_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.1
            )
            llm_with_tools = llm.bind_tools(tools)
            
            messages = [
                SystemMessage(content=AGENT_SYSTEM_PROMPT),
                HumanMessage(content=query)
            ]

            # Agent step 1: LLM decides tool call
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            def _clean_content(msg_content: Any) -> str:
                if isinstance(msg_content, str):
                    return msg_content
                if isinstance(msg_content, list):
                    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in msg_content)
                return str(msg_content)

            if ai_msg.tool_calls:
                step_idx = 1
                for tc in ai_msg.tool_calls:
                    t_name = tc["name"]
                    t_args = tc.get("args", {})
                    tools_used.append(t_name)

                    selected_tool = tool_map.get(t_name)
                    if selected_tool:
                        # Invoke selected tool
                        tool_res = selected_tool.invoke(t_args)
                    else:
                        tool_res = f"Tool '{t_name}' not found."

                    steps.append(
                        AgentExecutionStep(
                            step_number=step_idx,
                            thought=f"Decided to call '{t_name}' with parameters: {t_args}",
                            tool_name=t_name,
                            tool_input=json.dumps(t_args),
                            tool_output=str(tool_res)[:500]
                        )
                    )
                    step_idx += 1

                    messages.append(ToolMessage(tool_call_id=tc["id"], content=str(tool_res)))

                # Final synthesis step
                final_ai_msg = llm.invoke(messages)
                return AgentResponse(
                    query=query,
                    final_answer=_clean_content(final_ai_msg.content),
                    tools_used=list(set(tools_used)),
                    steps=steps,
                    total_steps=len(steps)
                )
            else:
                # Direct answer without tools
                return AgentResponse(
                    query=query,
                    final_answer=_clean_content(ai_msg.content),
                    tools_used=[],
                    steps=[],
                    total_steps=0
                )
        except Exception as e:
            logger.warning(f"LangChain Tool-Calling Agent error: {e}. Falling back to deterministic agent orchestrator.")

    # 2. Intelligent Deterministic Agent Orchestrator (Fallback for zero-cost offline testing)
    q_lower = query.lower()
    if any(w in q_lower for w in ["how many", "stats", "count", "list files", "documents uploaded", "files uploaded"]):
        # Stats Tool Selected
        chosen_tool = "get_document_stats"
        tool_out = tool_stats.invoke({})
        steps.append(
            AgentExecutionStep(
                step_number=1,
                thought="Detected user query regarding knowledge base statistics/metadata. Selecting 'get_document_stats'.",
                tool_name=chosen_tool,
                tool_input="{}",
                tool_output=tool_out
            )
        )
        return AgentResponse(
            query=query,
            final_answer=tool_out,
            tools_used=[chosen_tool],
            steps=steps,
            total_steps=1
        )
    elif any(w in q_lower for w in ["internet", "web", "latest news", "google", "online"]):
        # Web Search Fallback Selected
        chosen_tool = "web_search_fallback"
        tool_out = tool_web.invoke({"query": query})
        steps.append(
            AgentExecutionStep(
                step_number=1,
                thought="Detected query requiring external live information. Invoking 'web_search_fallback'.",
                tool_name=chosen_tool,
                tool_input=json.dumps({"query": query}),
                tool_output=tool_out[:500]
            )
        )
        return AgentResponse(
            query=query,
            final_answer=f"External Web Results for '{query}':\n\n{tool_out}",
            tools_used=[chosen_tool],
            steps=steps,
            total_steps=1
        )
    else:
        # Knowledge Base Search Selected
        chosen_tool = "search_knowledge_base"
        tool_out = tool_kb.invoke({"query": query})
        steps.append(
            AgentExecutionStep(
                step_number=1,
                thought="Determined query is about enterprise documents. Querying vector store using 'search_knowledge_base'.",
                tool_name=chosen_tool,
                tool_input=json.dumps({"query": query}),
                tool_output=tool_out[:500]
            )
        )
        return AgentResponse(
            query=query,
            final_answer=f"Synthesized from Enterprise Knowledge Base:\n\n{tool_out}",
            tools_used=[chosen_tool],
            steps=steps,
            total_steps=1
        )
