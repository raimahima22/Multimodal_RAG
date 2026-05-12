import sys
import gc
import os
import json
from datetime import datetime

import torch
import gradio as gr

from src.utils import clear_page_cache
from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator

HISTORY_FILE = "chat_history.json"


# =========================================================
# SAVE CHAT HISTORY
# =========================================================
def save_to_history(query, source_input, answer):

    history = []

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []

    history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "source": source_input,
            "answer": answer,
        }
    )

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# =========================================================
# MEMORY CLEANUP
# =========================================================
def aggressive_cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# MAIN
# =========================================================
def main(force_reindex: bool = False):

    print("\n🧠 Initializing Multimodal RAG System...\n")

    # -----------------------------------------------------
    # INITIALIZE COMPONENTS
    # -----------------------------------------------------
    indexer = MultimodalIndexer(force_recreate=force_reindex)

    retriever = MultimodalRetriever(indexer)

    generator = MultimodalGenerator()

    # -----------------------------------------------------
    # MODEL WARMUP
    # -----------------------------------------------------
    print("🔥 Warming up retrieval model...")

    _ = retriever._extract_text_embedding("warmup query")

    print("✅ System Ready!\n")

    # -----------------------------------------------------
    # INDEXING
    # -----------------------------------------------------
    print("📚 Checking document index...\n")

    if force_reindex or indexer.is_collection_empty():

        print("🔄 Indexing documents...\n")

        indexer.index_all_data("data")

        print("\n✅ Indexing completed!\n")

    else:
        print("✅ Existing index found. Skipping indexing.\n")

    # -----------------------------------------------------
    # SOURCE OPTIONS
    # -----------------------------------------------------
    source_options = {
        "Both Documents": None,
        "SBC": "data/sbc.pdf",
        "SPD": "data/spd.pdf",
    }

    # =====================================================
    # ANSWER QUERY
    # =====================================================
    def answer_query(message, history, source_choice):

        if not message or not message.strip():
            return history, ""

        query = message.strip()

        # -------------------------------------------------
        # SOURCE FILTER
        # -------------------------------------------------
        if source_choice.lower() in ["sbc", "sbc.pdf"]:
            source_filter = "data/sbc.pdf"

        elif source_choice.lower() in ["spd", "spd.pdf"]:
            source_filter = "data/spd.pdf"

        else:
            source_filter = None

        try:

            # -------------------------------------------------
            # RETRIEVAL
            # -------------------------------------------------
            hits = retriever.search(
                query=query,
                top_k=3,
                source_filter=source_filter,
            )

            # -------------------------------------------------
            # NO RESULTS
            # -------------------------------------------------
            if not hits:

                bot_response = """
❌ No relevant information found in the selected documents.
"""

            else:

                # -------------------------------------------------
                # SORT RESULTS
                # -------------------------------------------------
                context_hits = sorted(
                    hits,
                    key=lambda x: x.score,
                    reverse=True
                )

                best_hit = context_hits[0]

                source = best_hit.payload.get(
                    "source",
                    "Unknown"
                )

                page = best_hit.payload.get(
                    "page_number",
                    "N/A"
                )

                # -------------------------------------------------
                # GENERATE ANSWER
                # -------------------------------------------------
                answer = generator.generate_answer(
                    query,
                    context_hits
                )

                # -------------------------------------------------
                # FINAL RESPONSE
                # -------------------------------------------------
                bot_response = f"""
## 📄 Retrieved Context

- **Source:** `{os.path.basename(source)}`
- **Page:** `{page}`
- **Search Scope:** `{source_choice}`

---

## 🤖 Answer

{answer}
"""

                # -------------------------------------------------
                # SAVE HISTORY
                # -------------------------------------------------
                save_to_history(
                    query=query,
                    source_input=source_choice,
                    answer=answer
                )

            # -------------------------------------------------
            # APPEND USER MESSAGE
            # -------------------------------------------------
            history.append(
                {
                    "role": "user",
                    "content": query
                }
            )

            # -------------------------------------------------
            # APPEND ASSISTANT MESSAGE
            # -------------------------------------------------
            history.append(
                {
                    "role": "assistant",
                    "content": bot_response
                }
            )

            return history, ""

        except Exception as e:

            error_message = f"""
❌ Error while generating response:

```python
{str(e)}