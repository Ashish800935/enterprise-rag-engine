import pytest
from pydantic import ValidationError
from src.api.schemas import (
    QueryRequest,
    StructuredCitation,
    StructuredQueryResponse,
    AgentQueryResponse,
    AgentStepSchema
)

def test_query_request_validation():
    # Valid payload
    req = QueryRequest(query="What is HNSW?", top_k=4, use_hybrid=True)
    assert req.query == "What is HNSW?"
    assert req.top_k == 4
    assert req.use_hybrid is True

    # Invalid: query too short (< 2 chars)
    with pytest.raises(ValidationError):
        QueryRequest(query="a")

    # Invalid: top_k out of range (> 10)
    with pytest.raises(ValidationError):
        QueryRequest(query="Valid query", top_k=15)

def test_structured_response_serialization():
    citation = StructuredCitation(
        filename="report.pdf",
        chunk_index=2,
        exact_quote="Revenue grew 14% year over year."
    )
    res = StructuredQueryResponse(
        query="What was the growth?",
        answer="Revenue grew by 14%.",
        confidence_score=0.92,
        citations=[citation],
        suggested_followups=["What were Q4 projections?"],
        retrieval_latency_ms=11.5,
        total_latency_ms=620.0
    )
    assert res.confidence_score == 0.92
    assert len(res.citations) == 1
    assert res.citations[0].filename == "report.pdf"

def test_agent_telemetry_schema():
    step = AgentStepSchema(
        step_number=1,
        thought="Checking total documents in knowledge base",
        tool_name="get_document_stats",
        tool_input="{}",
        tool_output="Total documents: 3"
    )
    agent_res = AgentQueryResponse(
        query="Status check",
        final_answer="3 documents are indexed.",
        tools_used=["get_document_stats"],
        steps=[step],
        total_steps=1,
        latency_ms=45.2
    )
    assert agent_res.total_steps == 1
    assert agent_res.tools_used[0] == "get_document_stats"
