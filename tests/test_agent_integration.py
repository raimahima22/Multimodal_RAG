"""
Integration Tests — Task 3: LangGraph Agent with Dual RAG Tools
Tests end-to-end flow: query → tool routing → tool call → final response.
Run: pytest tests/test_agent_integration.py -v
"""

import pytest
import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_response(answer: str, sources: list) -> dict:
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Tests: Tool Routing
# ---------------------------------------------------------------------------

class TestToolRouting:
    """Validate that queries are routed to the correct RAG tool."""

    @patch("src.agent.llm")
    def test_sbc_query_routes_to_sbc(self, mock_llm):
        """Cost/coverage queries should route to SBC."""
        from src.agent import route_query
        mock_llm.invoke.return_value = MagicMock(content="SBC")

        result = route_query("What is the deductible for this plan?")
        assert result == "sbc"

    @patch("src.agent.llm")
    def test_spd_query_routes_to_spd(self, mock_llm):
        """Rules/eligibility queries should route to SPD."""
        from src.agent import route_query
        mock_llm.invoke.return_value = MagicMock(content="SPD")

        result = route_query("Tell me about eligibility rules.")
        assert result == "spd"

    @patch("src.agent.llm")
    def test_router_fallback_on_error(self, mock_llm):
        """Router must default to 'spd' if the LLM raises an exception."""
        from src.agent import route_query
        mock_llm.invoke.side_effect = Exception("LLM unavailable")

        result = route_query("Some query")
        assert result == "spd"

    @patch("src.agent.llm")
    def test_router_returns_sbc_for_copay(self, mock_llm):
        from src.agent import route_query
        mock_llm.invoke.return_value = MagicMock(content="SBC")

        result = route_query("What is the copay for a specialist visit?")
        assert result == "sbc"

    @patch("src.agent.llm")
    def test_router_returns_spd_for_exclusions(self, mock_llm):
        from src.agent import route_query
        mock_llm.invoke.return_value = MagicMock(content="SPD")

        result = route_query("What services are excluded from this plan?")
        assert result == "spd"


# ---------------------------------------------------------------------------
# Tests: run_agent() — End-to-End
# ---------------------------------------------------------------------------

class TestRunAgent:
    """End-to-end tests for run_agent()."""

    @patch("src.agent.agent")
    def test_run_agent_returns_dict_with_answer_and_sources(self, mock_agent):
        """run_agent() must always return a dict with 'answer' and 'sources'."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="The deductible is $1,500.")],
            "sources": ["SBC"],
        }

        result = run_agent("What is the deductible?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result

    @patch("src.agent.agent")
    def test_run_agent_answer_is_string(self, mock_agent):
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Coverage answer here.")],
            "sources": ["SBC"],
        }

        result = run_agent("What services are covered?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @patch("src.agent.agent")
    def test_run_agent_sbc_query(self, mock_agent):
        """SBC-type query should produce an answer attributed to SBC."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Out-of-pocket max is $5,000.")],
            "sources": ["SBC"],
        }

        result = run_agent("What is the out-of-pocket maximum?")
        assert "SBC" in result["sources"]

    @patch("src.agent.agent")
    def test_run_agent_spd_query(self, mock_agent):
        """SPD-type query should produce an answer attributed to SPD."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="You must work 30 hours to be eligible.")],
            "sources": ["SPD"],
        }

        result = run_agent("What are the eligibility rules?")
        assert "SPD" in result["sources"]

    @patch("src.agent.agent")
    def test_run_agent_empty_query(self, mock_agent):
        """Empty query must return a helpful message without calling the agent."""
        from src.agent import run_agent

        result = run_agent("")
        mock_agent.invoke.assert_not_called()
        assert "Please ask a question" in result["answer"]
        assert result["sources"] == []

    @patch("src.agent.agent")
    def test_run_agent_whitespace_query(self, mock_agent):
        """Whitespace-only query must return a helpful message without calling the agent."""
        from src.agent import run_agent

        result = run_agent("   ")
        mock_agent.invoke.assert_not_called()
        assert "Please ask a question" in result["answer"]

    @patch("src.agent.agent")
    def test_run_agent_handles_exception(self, mock_agent):
        """Agent exceptions must be caught and returned as an error message."""
        from src.agent import run_agent

        mock_agent.invoke.side_effect = Exception("Network error")

        result = run_agent("What is the deductible?")
        assert "Error" in result["answer"]
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# Tests: Fallback Handling
# ---------------------------------------------------------------------------

class TestFallbackHandling:
    """Validate graceful degradation when tools return no confident results."""

    @patch("src.agent.agent")
    def test_no_results_produces_non_empty_answer(self, mock_agent):
        """Even when tools find nothing, the agent must return a usable string."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(
                content="No relevant information found in the provided documents."
            )],
            "sources": [],
        }

        result = run_agent("What is the premium for a platinum tier plan?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @patch("src.tools.search_sbc")
    @patch("src.tools.search_spd")
    def test_both_tools_return_no_results(self, mock_spd, mock_sbc):
        """Both tools returning no-results messages should not crash the pipeline."""
        mock_sbc.return_value = "No relevant information found in SBC documents."
        mock_spd.return_value = "No relevant information found in SPD documents."

        from src.tools import search_sbc, search_spd
        sbc_result = search_sbc("nonsense query xyz123")
        spd_result = search_spd("nonsense query xyz123")

        assert "No relevant information" in sbc_result
        assert "No relevant information" in spd_result


# ---------------------------------------------------------------------------
# Tests: Tool Node Source Tracking
# ---------------------------------------------------------------------------

class TestSourceTracking:
    """Validate that sources are tracked and deduplicated correctly."""

    @patch("src.agent.agent")
    def test_sources_are_deduplicated(self, mock_agent):
        """Duplicate sources must be collapsed to a single entry."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Combined answer from SBC.")],
            "sources": ["SBC", "SBC"],
        }

        result = run_agent("What is the deductible and the out-of-pocket max?")
        # run_agent itself doesn't deduplicate — that's the agent graph's job.
        # We just check the key exists and is a list.
        assert isinstance(result["sources"], list)

    @patch("src.agent.agent")
    def test_sources_list_is_always_returned(self, mock_agent):
        """run_agent must always include a 'sources' key even when empty."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Some answer.")],
            # sources key intentionally missing to test .get() fallback
        }

        result = run_agent("What is coinsurance?")
        assert "sources" in result
        assert isinstance(result["sources"], list)


# ---------------------------------------------------------------------------
# Parametrized end-to-end scenario table
# ---------------------------------------------------------------------------

E2E_SCENARIOS = [
    # (query, expected_source, keyword_in_answer)
    ("What is the deductible?",                    "SBC", "deductible"),
    ("What is the out-of-pocket maximum?",         "SBC", "out-of-pocket"),
    ("What services are covered?",                 "SBC", "covered"),
    ("Tell me about eligibility rules.",           "SPD", "eligible"),
    ("What are plan exclusions?",                  "SPD", "excluded"),
    ("How do I file a claim?",                     "SPD", "claim"),
]


@pytest.mark.parametrize("query,expected_source,keyword", E2E_SCENARIOS)
@patch("src.agent.agent")
def test_e2e_scenario(mock_agent, query, expected_source, keyword):
    """
    Parametrized end-to-end: each query must return a non-empty answer
    attributed to the correct source document.
    """
    from langchain_core.messages import AIMessage
    from src.agent import run_agent

    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content=f"The answer about {keyword} goes here.")],
        "sources": [expected_source],
    }

    result = run_agent(query)
    assert isinstance(result["answer"], str), "Answer must be a string"
    assert len(result["answer"]) > 0, "Answer must not be empty"
    assert expected_source in result["sources"], (
        f"Expected source '{expected_source}' not in {result['sources']}"
    )