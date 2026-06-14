"""
Integration Tests — Task 3: LangGraph Agent with Dual RAG Tools
Tests end-to-end flow: query → routing → tool call(s) → final response.

Run:  pytest tests/test_agent_integration.py -v
"""

import pytest
import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Tests: route_query()
# ---------------------------------------------------------------------------

class TestToolRouting:
    """Validate that queries are routed to the correct RAG tool(s)."""

    @patch("src.agent.llm")
    def test_sbc_keyword_routes_to_sbc(self, mock_llm):
        """Deductible query should route to SBC via keyword shortcut (no LLM call)."""
        from src.agent import route_query
        result = route_query("What is the deductible for this plan?")
        mock_llm.invoke.assert_not_called()   # keyword match — LLM not needed
        assert result == "sbc"

    @patch("src.agent.llm")
    def test_spd_keyword_routes_to_spd(self, mock_llm):
        """Eligibility query should route to SPD via keyword shortcut."""
        from src.agent import route_query
        result = route_query("Tell me about eligibility rules.")
        mock_llm.invoke.assert_not_called()
        assert result == "spd"

    @patch("src.agent.llm")
    def test_both_keywords_routes_to_both(self, mock_llm):
        """Query with keywords from both categories should route to BOTH."""
        from src.agent import route_query
        result = route_query("What is the deductible and am I eligible?")
        assert result == "both"

    @patch("src.agent.llm")
    def test_router_llm_fallback_sbc(self, mock_llm):
        """Ambiguous query with LLM returning SBC."""
        from src.agent import route_query
        mock_llm.invoke.return_value = MagicMock(content="SBC")
        result = route_query("What happens when I visit an urgent care center?")
        assert result == "sbc"

    @patch("src.agent.llm")
    def test_router_llm_fallback_both(self, mock_llm):
        """Ambiguous query where LLM returns BOTH."""
        from src.agent import route_query
        mock_llm.invoke.return_value = MagicMock(content="BOTH")
        result = route_query("Give me a full overview of my plan.")
        assert result == "both"

    @patch("src.agent.llm")
    def test_router_exception_defaults_to_spd(self, mock_llm):
        """LLM exception must default to SPD (the more comprehensive document)."""
        from src.agent import route_query
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        result = route_query("Something unclear")
        assert result == "spd"

    @patch("src.agent.llm")
    def test_copay_routes_to_sbc(self, mock_llm):
        from src.agent import route_query
        result = route_query("What is the copay for a specialist visit?")
        assert result == "sbc"

    @patch("src.agent.llm")
    def test_exclusions_routes_to_spd(self, mock_llm):
        from src.agent import route_query
        result = route_query("What services are excluded from this plan?")
        assert result == "spd"


# ---------------------------------------------------------------------------
# Tests: run_agent() — end-to-end
# ---------------------------------------------------------------------------

class TestRunAgent:
    """End-to-end tests for run_agent()."""

    @patch("src.agent.agent")
    def test_returns_dict_with_required_keys(self, mock_agent):
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="The deductible is $1,500.")],
            "sources": ["SBC"],
            "routing": "sbc",
            "token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }

        result = run_agent("What is the deductible?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "token_usage" in result

    @patch("src.agent.agent")
    def test_answer_is_non_empty_string(self, mock_agent):
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Coverage answer here.")],
            "sources": ["SBC"],
            "routing": "sbc",
            "token_usage": None,
        }

        result = run_agent("What services are covered?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @patch("src.agent.agent")
    def test_sbc_query_attributed_to_sbc(self, mock_agent):
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Out-of-pocket max is $5,000.")],
            "sources": ["SBC"],
            "routing": "sbc",
            "token_usage": None,
        }

        result = run_agent("What is the out-of-pocket maximum?")
        assert "SBC" in result["sources"]

    @patch("src.agent.agent")
    def test_spd_query_attributed_to_spd(self, mock_agent):
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="You must work 30 hours to be eligible.")],
            "sources": ["SPD"],
            "routing": "spd",
            "token_usage": None,
        }

        result = run_agent("What are the eligibility rules?")
        assert "SPD" in result["sources"]

    @patch("src.agent.agent")
    def test_both_tool_query_has_both_sources(self, mock_agent):
        """A hybrid query should surface both SBC and SPD in sources."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Deductible is $1,500. You must work 30h/week.")],
            "sources": ["SBC", "SPD"],
            "routing": "both",
            "token_usage": None,
        }

        result = run_agent("What is the deductible and am I eligible?")
        assert "SBC" in result["sources"]
        assert "SPD" in result["sources"]

    @patch("src.agent.agent")
    def test_empty_query_bypasses_agent(self, mock_agent):
        from src.agent import run_agent

        result = run_agent("")
        mock_agent.invoke.assert_not_called()
        assert "Please ask a question" in result["answer"]
        assert result["sources"] == []
        assert result["token_usage"] is None

    @patch("src.agent.agent")
    def test_whitespace_query_bypasses_agent(self, mock_agent):
        from src.agent import run_agent

        result = run_agent("   ")
        mock_agent.invoke.assert_not_called()
        assert "Please ask a question" in result["answer"]

    @patch("src.agent.agent")
    def test_exception_returns_error_message(self, mock_agent):
        from src.agent import run_agent

        mock_agent.invoke.side_effect = Exception("Network error")
        result = run_agent("What is the deductible?")
        assert "error" in result["answer"].lower()
        assert result["sources"] == []

    @patch("src.agent.agent")
    def test_sources_are_deduplicated(self, mock_agent):
        """Duplicate sources in state must be collapsed by run_agent()."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Combined SBC answer.")],
            "sources": ["SBC", "SBC"],
            "routing": "sbc",
            "token_usage": None,
        }

        result = run_agent("What is the deductible and the out-of-pocket max?")
        assert result["sources"].count("SBC") == 1, "Duplicate SBC source not deduplicated"

    @patch("src.agent.agent")
    def test_sources_key_always_present(self, mock_agent):
        """sources must always be a list even when the key is missing from state."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Some answer.")],
            # 'sources' key intentionally omitted
        }

        result = run_agent("What is coinsurance?")
        assert "sources" in result
        assert isinstance(result["sources"], list)

    @patch("src.agent.agent")
    def test_token_usage_is_returned(self, mock_agent):
        """token_usage dict must be passed through from agent state."""
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Answer.")],
            "sources": ["SBC"],
            "token_usage": {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        }

        result = run_agent("What is the copay?")
        assert result["token_usage"] is not None
        assert result["token_usage"]["total_tokens"] == 280


# ---------------------------------------------------------------------------
# Tests: Fallback handling
# ---------------------------------------------------------------------------

class TestFallbackHandling:

    @patch("src.agent.agent")
    def test_no_results_returns_non_empty_answer(self, mock_agent):
        from langchain_core.messages import AIMessage
        from src.agent import run_agent

        mock_agent.invoke.return_value = {
            "messages": [AIMessage(
                content="No relevant information found in the provided documents."
            )],
            "sources": [],
            "routing": "sbc",
            "token_usage": None,
        }

        result = run_agent("What is the premium for a platinum tier plan?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @patch("src.tools.search_sbc")
    @patch("src.tools.search_spd")
    def test_both_tools_returning_no_results_does_not_crash(self, mock_spd, mock_sbc):
        mock_sbc.return_value = "No relevant information found in SBC documents."
        mock_spd.return_value = "No relevant information found in SPD documents."

        from src.tools import search_sbc, search_spd
        assert "No relevant information" in search_sbc("nonsense xyz123")
        assert "No relevant information" in search_spd("nonsense xyz123")

    def test_is_empty_result_detects_no_result(self):
        from src.agent import _is_empty_result
        assert _is_empty_result("No relevant information found")
        assert _is_empty_result("Answer not found in provided documents")
        assert not _is_empty_result("The deductible is $1,500.")

    def test_is_vague_answer_detects_refusal(self):
        from src.agent import _is_vague_answer
        assert _is_vague_answer("I don't know the answer to that question.")
        assert _is_vague_answer("I am unable to answer this query.")
        assert not _is_vague_answer("The copay for a specialist is $40.")


# ---------------------------------------------------------------------------
# Tests: final_answer_node merging
# ---------------------------------------------------------------------------

class TestFinalAnswerNode:

    def _make_state(self, tool_results: list, sources: list):
        """Build a minimal AgentState for testing final_answer_node."""
        from langchain_core.messages import ToolMessage
        messages = []
        for i, r in enumerate(tool_results):
            messages.append(ToolMessage(
                content=json.dumps(r),
                tool_call_id=f"call_{i}",
            ))
        return {"messages": messages, "sources": sources, "routing": "both", "token_usage": None}

    def test_single_result_returns_clean_answer(self):
        from src.agent import final_answer_node
        state = self._make_state(
            [{"answer": "The deductible is $1,500.", "source": "SBC"}],
            ["SBC"]
        )
        result = final_answer_node(state)
        assert "1,500" in result["messages"][-1].content

    def test_two_results_are_merged(self):
        from src.agent import final_answer_node
        state = self._make_state(
            [
                {"answer": "The deductible is $1,500.", "source": "SBC"},
                {"answer": "You must work 30 hours per week.", "source": "SPD"},
            ],
            ["SBC", "SPD"]
        )
        content = final_answer_node(state)["messages"][-1].content
        assert "1,500" in content
        assert "30 hours" in content

    def test_all_empty_results_returns_fallback(self):
        from src.agent import final_answer_node, _FALLBACK_MESSAGE
        state = self._make_state(
            [
                {"answer": "No relevant information found in SBC documents.", "source": "SBC"},
                {"answer": "No relevant information found in SPD documents.", "source": "SPD"},
            ],
            []
        )
        content = final_answer_node(state)["messages"][-1].content
        assert content == _FALLBACK_MESSAGE

    def test_partial_empty_adds_note(self):
        from src.agent import final_answer_node
        state = self._make_state(
            [
                {"answer": "The deductible is $1,500.", "source": "SBC"},
                {"answer": "No relevant information found in SPD documents.", "source": "SPD"},
            ],
            ["SBC"]
        )
        content = final_answer_node(state)["messages"][-1].content
        assert "1,500" in content
        assert "SPD" in content   # note about the empty result


# ---------------------------------------------------------------------------
# Parametrized end-to-end scenarios
# ---------------------------------------------------------------------------

E2E_SCENARIOS = [
    ("What is the deductible?",            "SBC", "deductible"),
    ("What is the out-of-pocket maximum?", "SBC", "out-of-pocket"),
    ("What services are covered?",         "SBC", "covered"),
    ("Tell me about eligibility rules.",   "SPD", "eligible"),
    ("What are plan exclusions?",          "SPD", "excluded"),
    ("How do I file a claim?",             "SPD", "claim"),
]


@pytest.mark.parametrize("query,expected_source,keyword", E2E_SCENARIOS)
@patch("src.agent.agent")
def test_e2e_scenario(mock_agent, query, expected_source, keyword):
    from langchain_core.messages import AIMessage
    from src.agent import run_agent

    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content=f"The answer about {keyword} goes here.")],
        "sources": [expected_source],
        "routing": expected_source.lower(),
        "token_usage": None,
    }

    result = run_agent(query)
    assert isinstance(result["answer"], str), "Answer must be a string"
    assert len(result["answer"]) > 0, "Answer must not be empty"
    assert expected_source in result["sources"], (
        f"Expected '{expected_source}' not in {result['sources']}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    from pathlib import Path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("test_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    report = results_dir / f"agent_integration_{timestamp}.txt"
    subprocess.run(f"pytest {__file__} -v 2>&1 | tee {report}", shell=True)
    print(f"\nResults saved → {report}")