# Healthcare Benefits Voice Assistant

A voice-enabled AI agent that answers questions about healthcare benefit documents using a Multimodal Retrieval-Augmented Generation (RAG) pipeline backed by Qdrant and ColQwen2.5 multimodal embeddings.

---

## Architecture

```
Microphone / audio file
        ↓
  faster-whisper (STT)
        ↓
  LangGraph Agent
   ├── route_query() → SBC | SPD | BOTH
   ├── search_sbc_tool  →  ColQwen2.5 + Qdrant (sbc_collection)
   └── search_spd_tool  →  ColQwen2.5 + Qdrant (spd_collection)
        ↓
  Llama 4 Scout (via OpenRouter)
        ↓
  Piper TTS
        ↓
  Gradio UI (answer text + spoken audio + latency metrics)
```

```
                ┌─────────────────┐
                │   Documents     │
                │ PDFs / Images   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │      
                │   Chunking      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Visual Embedding│
                │  ColQwen2.5     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Qdrant VectorDB │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Hybrid Retrieval│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Vision-Language │
                │  Llama 4 Scout  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Final Answer   │
                │ + Piper TTS     │
                └─────────────────┘
```

**Document types**

| Abbreviation | Full name | Contents |
|---|---|---|
| SBC | Summary of Benefits and Coverage | Costs, deductibles, copays, coinsurance, OOP maxima |
| SPD | Summary Plan Description | Eligibility, exclusions, claims, appeals, definitions, legal |

---

## Features

### Document Ingestion

Supports ingestion of PDFs, scanned documents, images, OCR-heavy files, and mixed visual-text documents.

### Patch-Based Visual Chunking

Documents are segmented into overlapping 512px visual patches with a 160px stride, improving localized retrieval, page-region understanding, and multimodal relevance scoring.

### Multimodal Embeddings

ColQwen2.5 generates multi-vector embeddings for each patch, combining visual and textual signal for richer semantic search.

### Vector Database Integration

All embeddings, OCR text, and metadata are stored in Qdrant for efficient similarity search and scalable retrieval across separate `sbc_collection` and `spd_collection` namespaces.

### Hybrid Retrieval Pipeline

Combines semantic vector search, OCR-aware retrieval, and multimodal similarity matching. Retrieved chunks are reranked using OCR-sensitive scoring and aggregated at the page level for contextual coherence.

### Vision-Language Answer Generation

Uses Llama 4 Scout (via OpenRouter) to generate grounded answers from retrieved multimodal context.

### Voice Interface

End-to-end voice pipeline: faster-whisper for STT, Piper TTS for streaming speech synthesis, and a Gradio UI with a live latency metrics strip.

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding Model | ColQwen2.5 |
| Vector Database | Qdrant |
| OCR Engine | Tesseract OCR |
| LLM | Llama 4 Scout (via OpenRouter) |
| Agent Framework | LangGraph |
| STT | faster-whisper |
| TTS | Piper |
| UI | Gradio |
| PDF Processing | pdf2image / Poppler |
| Image Processing | PIL / Pillow |
| Deep Learning | PyTorch |

---

## Requirements

- Python 3.12
- CUDA GPU recommended (CPU works but is slower for embedding generation)
- Tesseract OCR: `sudo apt install tesseract-ocr`
- Poppler: `sudo apt install poppler-utils`

---

## Installation

```bash
# 1. Clone the repository
git clone <repository_url>
cd healthcare_agent

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your keys (see below)
```

### Environment variables

```env
OPENROUTER_API_KEY=your_openrouter_key_here
PIPER_MODEL_PATH=models/en_US-amy-medium.onnx   # path to your Piper voice model
QDRANT_PATH=./qdrant_db                          # where Qdrant stores its data
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | API key for OpenRouter (Llama 4 Scout) |
| `PIPER_MODEL_PATH` | No | `models/en_US-amy-medium.onnx` | Path to Piper ONNX model |
| `QDRANT_PATH` | No | `./qdrant_db` | Local path for Qdrant persistent storage |

If running in Google Colab, you can load the `.env` from Google Drive:

```python
load_dotenv('/content/drive/MyDrive/.env')
```

### Download the Piper voice model

```bash
mkdir -p models
wget -O models/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget -O models/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

---

## Data layout

Place your PDF documents in the following folders before ingestion:

```
data/
├── sbc/
│   ├── plan_a_sbc.pdf
│   └── plan_b_sbc.pdf
└── spd/
    ├── plan_a_spd.pdf
    └── ...
```

---

## Usage

### Indexing documents via Python API

```python
from src.indexer import MultimodalIndexer

indexer = MultimodalIndexer()
indexer.index_all_data("data")
```

### Retrieving relevant chunks

```python
from src.retriever import MultimodalRetriever

retriever = MultimodalRetriever(indexer)
results = retriever.search("What is the deductible?")
```

### Generating answers

```python
from src.generator import MultimodalGenerator

generator = MultimodalGenerator()
answer = generator.generate_answer(query, results)
print(answer)
```

---

## Ingestion

### SBC ingestion

Ingests all PDFs from `data/sbc/` into `sbc_collection` in Qdrant.

```bash
python -m src.ingest_sbc            # first run
python -m src.ingest_sbc --reindex  # force rebuild
```

What happens during ingestion:
1. Each PDF page is converted to a high-DPI image.
2. Overlapping 512px patches are extracted with a 160px stride.
3. Tesseract OCR extracts text at both page and patch level.
4. ColQwen2.5 generates multi-vector embeddings for each patch.
5. Embeddings, OCR text, and metadata are stored in `sbc_collection`.

### SPD ingestion

Identical pipeline, targeting `spd_collection`.

```bash
python -m src.ingest_spd
python -m src.ingest_spd --reindex
```

---

## LangGraph Agent

The agent is a three-node LangGraph graph:

```
agent_node → tools_node → final_answer_node
```

**Routing logic** (`route_query`):

1. Keyword pre-scan — fast, no LLM call needed for clear-cut queries.
2. If ambiguous, a lightweight LLM call classifies as SBC, SPD, or BOTH.
3. On LLM error, defaults to SPD (the more comprehensive document).

**Routing categories:**

| Route | Triggered by |
|---|---|
| `sbc` | deductible, copay, coinsurance, OOP, cost, premium, coverage summary |
| `spd` | eligibility, exclusion, claim, appeal, HIPAA, PPO, definition, policy |
| `both` | query contains keywords from both categories |

**Fallback handling — three tiers:**

1. Tool returns an empty-result phrase → `_FALLBACK_MESSAGE` is shown.
2. Tool raises an exception → caught, logged, graceful error returned.
3. LLM produces a vague non-answer → detected and replaced with fallback.

---

## Voice Interface

```bash
python main.py
```

The app is available at `http://localhost:7860`. A public share link is also printed on startup.

### Metrics strip

Every response displays:

| Metric | Description |
|---|---|
| STT | Time from audio file to transcribed text |
| Agent | Time from text query to final answer text |
| TTS | Time-to-first-chunk for speech synthesis |
| Total | Sum of all three |

### Observed latency (Colab T4 GPU)

| Stage | Typical range |
|---|---|
| STT (faster-whisper base.en, CPU) | 1.5 – 3.5 s |
| Agent (retrieval + Llama 4 Scout via OpenRouter) | 8 – 20 s |
| TTS (Piper, CPU) | 1 – 4 s (scales with answer length) |
| **End-to-end** | **~12 – 28 s** |

The dominant cost is the agent step. Reducing `top_k` in retrieval from 3 to 2 pages shaves ~2 s off the generator call.

---

## Testing

```bash
# All tests
pytest tests/ -v

# Individual suites
pytest tests/test_sbc_retrieval.py -v
pytest tests/test_spd_retrieval.py -v
pytest tests/test_agent_integration.py -v

# With HTML report
pytest tests/ -v --html=test_results/report.html
```

---

## Project structure

```
healthcare_agent/
├── data/
│   ├── sbc/              ← place SBC PDFs here
│   └── spd/              ← place SPD PDFs here
├── models/               ← Piper ONNX voice model
├── src/
│   ├── agent.py          ← LangGraph agent, routing, fallback logic
│   ├── generator.py      ← Multimodal LLM answer generation
│   ├── indexer.py        ← ColQwen2.5 + Qdrant ingestion
│   ├── retriever.py      ← Hybrid retrieval (embedding + BM25 + keyword)
│   ├── tools.py          ← search_sbc() and search_spd() tool functions
│   ├── utils.py          ← PDF/image helpers, caching
│   ├── voice.py          ← STT (faster-whisper) + TTS (Piper)
│   ├── ingest_sbc.py     ← SBC ingestion entry point
│   └── ingest_spd.py     ← SPD ingestion entry point
├── tests/
│   ├── test_sbc_retrieval.py
│   ├── test_spd_retrieval.py
│   └── test_agent_integration.py
├── main.py               ← Gradio app entry point
├── requirements.txt
├── .env.example
└── README.md
```