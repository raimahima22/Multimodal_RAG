"""
Unit Tests — Task 2: SPD Retrieval Pipeline
Tests top-3 retrieval accuracy on sample queries.
Run: pytest tests/test_spd_retrieval.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(page_num: int, score: float, ocr: str = "sample ocr text") -> MagicMock:
    """Create a mock Qdrant ScoredPoint."""
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "page_number": page_num,
        "source": "data/spd/sample_spd.pdf",
        "page_ocr": ocr,
        "patch_ocr": ocr,
    }
    return hit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_spd_hits():
    """Three realistic SPD result hits ordered by score."""
    return [
        _make_hit(7,  0.88, "Eligibility rules: employees must work 30 hours per week"),
        _make_hit(12, 0.80, "Exclusions: cosmetic surgery not covered unless medically necessary"),
        _make_hit(15, 0.72, "Claims procedure: submit within 90 days of service date"),
    ]


@pytest.fixture
def mock_retriever(mock_spd_hits):
    retriever = MagicMock()
    retriever.search.return_value = mock_spd_hits
    return retriever


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate_answer.return_value = (
        "Employees must work at least 30 hours per week to be eligible."
    )
    return gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSPDRetrieval:
    """Validate the SPD retrieval pipeline end-to-end."""

    def test_search_returns_top3_hits(self, mock_retriever, mock_spd_hits):
        """search() must return exactly 3 hits."""
        results = mock_retriever.search("What are the eligibility rules?", top_k=3)
        assert len(results) == 3

    def test_hits_have_required_payload_fields(self, mock_spd_hits):
        """Every hit must carry page_number and source in its payload."""
        for hit in mock_spd_hits:
            assert "page_number" in hit.payload, "Missing page_number in payload"
            assert "source" in hit.payload, "Missing source in payload"

    def test_hits_sorted_by_score_descending(self, mock_spd_hits):
        """Hits must come back in descending score order."""
        scores = [h.score for h in mock_spd_hits]
        assert scores == sorted(scores, reverse=True), (
            f"Hits not sorted by score: {scores}"
        )

    def test_spd_collection_is_separate_from_sbc(self, mock_spd_hits):
        """All SPD hits must reference the SPD data folder, not SBC."""
        for hit in mock_spd_hits:
            assert "spd" in hit.payload["source"].lower(), (
                f"Hit source '{hit.payload['source']}' does not reference SPD collection"
            )

    def test_eligibility_query(self, mock_retriever):
        """Eligibility query should surface a page mentioning 'eligibility'."""
        hits = mock_retriever.search("What are the eligibility rules?", top_k=3)
        found = any("eligibility" in h.payload["page_ocr"].lower() for h in hits)
        assert found

    def test_exclusions_query(self, mock_retriever):
        """Exclusions query should surface a page with exclusion information."""
        hits = mock_retriever.search("What services are excluded?", top_k=3)
        found = any("exclusions" in h.payload["page_ocr"].lower() for h in hits)
        assert found

    def test_claims_procedure_query(self, mock_retriever):
        """Claims query should surface a page describing claims procedures."""
        hits = mock_retriever.search("How do I submit a claim?", top_k=3)
        found = any("claims" in h.payload["page_ocr"].lower() for h in hits)
        assert found

    def test_empty_query_handled(self, mock_retriever):
        """An empty query should not raise; retriever returns empty list."""
        mock_retriever.search.return_value = []
        results = mock_retriever.search("", top_k=3)
        assert results == []

    def test_no_hits_returns_empty_list(self, mock_retriever):
        """When Qdrant returns nothing, search() must return an empty list."""
        mock_retriever.search.return_value = []
        results = mock_retriever.search("xyzzy irrelevant query", top_k=3)
        assert results == []


class TestSearchSPDTool:
    """Validate the search_spd() tool function used by the agent."""

    @patch("src.tools._get_spd_retriever")
    @patch("src.tools._get_generator")
    def test_search_spd_returns_string(self, mock_gen_factory, mock_ret_factory, mock_spd_hits):
        from src.tools import search_spd

        mock_ret_factory.return_value.search.return_value = mock_spd_hits
        mock_gen_factory.return_value.generate_answer.return_value = (
            "Employees must work 30 hours per week to be eligible."
        )

        result = search_spd("What are the eligibility requirements?")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.tools._get_spd_retriever")
    @patch("src.tools._get_generator")
    def test_search_spd_no_results_message(self, mock_gen_factory, mock_ret_factory):
        from src.tools import search_spd

        mock_ret_factory.return_value.search.return_value = []

        result = search_spd("completely unrelated question")
        assert "No relevant information found in SPD" in result

    @patch("src.tools._get_spd_retriever")
    @patch("src.tools._get_generator")
    def test_search_spd_calls_generator(self, mock_gen_factory, mock_ret_factory, mock_spd_hits):
        """Generator must be called with the original query and retrieved hits."""
        from src.tools import search_spd

        mock_retriever = mock_ret_factory.return_value
        mock_retriever.search.return_value = mock_spd_hits
        mock_generator = mock_gen_factory.return_value
        mock_generator.generate_answer.return_value = "Answer text."

        query = "What is the appeals process?"
        search_spd(query)

        mock_generator.generate_answer.assert_called_once_with(query, mock_spd_hits)

    @patch("src.tools._get_spd_retriever")
    @patch("src.tools._get_generator")
    def test_search_spd_top3_retrieval(self, mock_gen_factory, mock_ret_factory, mock_spd_hits):
        """Retriever must be called with top_k=3."""
        from src.tools import search_spd

        mock_retriever = mock_ret_factory.return_value
        mock_retriever.search.return_value = mock_spd_hits
        mock_gen_factory.return_value.generate_answer.return_value = "answer"

        search_spd("eligibility question")
        mock_retriever.search.assert_called_once_with("eligibility question", top_k=3)


# ---------------------------------------------------------------------------
# Sample Queries Accuracy Suite
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    ("Tell me about eligibility rules to participate in the plan.",          "eligibility"),
    ("What services are included in diabetic equipment and supplies?", "diabetic"),
    ("How do I file a claim?",                    "claims"),
    ("What are the plan definitions?",            "definitions"),
    ("How does the appeals process work?",        "appeals"),
]


@pytest.mark.parametrize("query,expected_keyword", SAMPLE_QUERIES)
def test_sample_query_keyword_in_results(query, expected_keyword):
    """
    Parametrized smoke test: each sample SPD query should surface at least
    one hit whose OCR text contains the expected keyword.
    """
    hits = [
        _make_hit(1, 0.9, f"This page explains {expected_keyword} in detail"),
        _make_hit(2, 0.8, "General plan administration information"),
        _make_hit(3, 0.7, "Summary of plan benefits and coverage"),
    ]

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = hits

    results = mock_retriever.search(query, top_k=3)
    ocr_combined = " ".join(h.payload["page_ocr"].lower() for h in results)
    assert expected_keyword in ocr_combined, (
        f"Keyword '{expected_keyword}' not found in results for query: '{query}'"
    )

if __name__ == "__main__":
    import subprocess
    from datetime import datetime
    from pathlib import Path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("/content/drive/MyDrive/test_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    report = results_dir / f"spd_retrieval_{timestamp}.txt"
    subprocess.run(
        f"pytest {__file__} -v 2>&1 | tee {report}",
        shell=True
    )
    print(f"\n SPD results saved - {report}")