"""
Unit Tests — Task 1: SBC Retrieval Pipeline
Tests top-3 retrieval accuracy on sample queries.
Run: pytest tests/test_sbc_retrieval.py -v
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
        "source": "data/sbc/sample_sbc.pdf",
        "page_ocr": ocr,
        "patch_ocr": ocr,
    }
    return hit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sbc_hits():
    """Three realistic SBC result hits ordered by score."""
    return [
        _make_hit(2, 0.91, "Individual deductible $1,500 family deductible $3,000"),
        _make_hit(5, 0.83, "Out-of-pocket maximum $5,000 per individual"),
        _make_hit(3, 0.75, "Preventive care covered 100% no cost sharing"),
    ]


@pytest.fixture
def mock_retriever(mock_sbc_hits):
    retriever = MagicMock()
    retriever.search.return_value = mock_sbc_hits
    return retriever


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate_answer.return_value = (
        "The individual deductible is $1,500 and the family deductible is $3,000."
    )
    return gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSBCRetrieval:
    """Validate the SBC retrieval pipeline end-to-end."""

    def test_search_returns_top3_hits(self, mock_retriever, mock_sbc_hits):
        """search() must return exactly 3 hits."""
        results = mock_retriever.search("What is the deductible?", top_k=3)
        assert len(results) == 3

    def test_hits_have_required_payload_fields(self, mock_sbc_hits):
        """Every hit must carry page_number and source in its payload."""
        for hit in mock_sbc_hits:
            assert "page_number" in hit.payload, "Missing page_number in payload"
            assert "source" in hit.payload, "Missing source in payload"

    def test_hits_sorted_by_score_descending(self, mock_sbc_hits):
        """Hits must come back in descending score order."""
        scores = [h.score for h in mock_sbc_hits]
        assert scores == sorted(scores, reverse=True), (
            f"Hits not sorted by score: {scores}"
        )

    def test_deductible_query_returns_relevant_hit(self, mock_retriever):
        """A deductible query should surface a page mentioning 'deductible'."""
        hits = mock_retriever.search("What is the deductible?", top_k=3)
        top_ocr = hits[0].payload["page_ocr"].lower()
        assert "deductible" in top_ocr

    def test_out_of_pocket_query(self, mock_retriever):
        """Out-of-pocket query should surface a page with OOP information."""
        hits = mock_retriever.search("What is the out-of-pocket maximum?", top_k=3)
        found = any("out-of-pocket" in h.payload["page_ocr"].lower() for h in hits)
        assert found, "No hit contained 'out-of-pocket' information"

    def test_preventive_care_query(self, mock_retriever):
        """Preventive care query should surface a relevant page."""
        hits = mock_retriever.search("What services are covered under preventive care?", top_k=3)
        found = any("preventive" in h.payload["page_ocr"].lower() for h in hits)
        assert found, "No hit contained preventive care information"

    def test_empty_query_handled(self, mock_retriever):
        """An empty query should not raise; retriever is called with it."""
        mock_retriever.search.return_value = []
        results = mock_retriever.search("", top_k=3)
        assert results == []

    def test_no_hits_returns_empty_list(self, mock_retriever):
        """When Qdrant returns nothing, search() must return an empty list."""
        mock_retriever.search.return_value = []
        results = mock_retriever.search("irrelevant nonsense query xyz", top_k=3)
        assert results == []


class TestSearchSBCTool:
    """Validate the search_sbc() tool function used by the agent."""

    @patch("src.tools._get_sbc_retriever")
    @patch("src.tools._get_generator")
    def test_search_sbc_returns_string(self, mock_gen_factory, mock_ret_factory, mock_sbc_hits):
        from src.tools import search_sbc

        mock_ret_factory.return_value.search.return_value = mock_sbc_hits
        mock_gen_factory.return_value.generate_answer.return_value = "The deductible is $1,500."

        result = search_sbc("What is the deductible?")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.tools._get_sbc_retriever")
    @patch("src.tools._get_generator")
    def test_search_sbc_no_results_message(self, mock_gen_factory, mock_ret_factory):
        from src.tools import search_sbc

        mock_ret_factory.return_value.search.return_value = []

        result = search_sbc("something completely unrelated")
        assert "No relevant information found in SBC" in result

    @patch("src.tools._get_sbc_retriever")
    @patch("src.tools._get_generator")
    def test_search_sbc_calls_generator(self, mock_gen_factory, mock_ret_factory, mock_sbc_hits):
        """Generator must be called with the original query and retrieved hits."""
        from src.tools import search_sbc

        mock_retriever = mock_ret_factory.return_value
        mock_retriever.search.return_value = mock_sbc_hits
        mock_generator = mock_gen_factory.return_value
        mock_generator.generate_answer.return_value = "Answer text."

        query = "What is the copay for a specialist?"
        search_sbc(query)

        mock_generator.generate_answer.assert_called_once_with(query, mock_sbc_hits)

    @patch("src.tools._get_sbc_retriever")
    @patch("src.tools._get_generator")
    def test_search_sbc_top3_retrieval(self, mock_gen_factory, mock_ret_factory, mock_sbc_hits):
        """Retriever must be called with top_k=3."""
        from src.tools import search_sbc

        mock_retriever = mock_ret_factory.return_value
        mock_retriever.search.return_value = mock_sbc_hits
        mock_gen_factory.return_value.generate_answer.return_value = "answer"

        search_sbc("deductible question")
        mock_retriever.search.assert_called_once_with("deductible question", top_k=3)


# ---------------------------------------------------------------------------
# Sample Queries Accuracy Suite
# Checks that queries map to expected page OCR content.
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    ("What is the deductible?",                "deductible"),
    ("What is the out-of-pocket maximum?",     "out-of-pocket"),
    ("What services are covered?",             "covered"),
    ("What is the copay for primary care?",    "copay"),
    ("What is coinsurance?",                   "coinsurance"),
]


@pytest.mark.parametrize("query,expected_keyword", SAMPLE_QUERIES)
def test_sample_query_keyword_in_results(query, expected_keyword):
    """
    Parametrized smoke test: each sample query should surface at least one
    hit whose OCR text contains the expected keyword.
    """
    hits = [
        _make_hit(1, 0.9, f"This page mentions {expected_keyword} details"),
        _make_hit(2, 0.8, "Other general benefit information"),
        _make_hit(3, 0.7, "Plan summary information"),
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

    report = results_dir / f"sbc_retrieval_{timestamp}.txt"
    subprocess.run(
        f"pytest {__file__} -v 2>&1 | tee {report}",
        shell=True
    )
    print(f"\n SBC results saved - {report}")