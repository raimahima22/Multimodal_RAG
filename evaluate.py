import pandas as pd
import time
from pathlib import Path
from datetime import datetime

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator


class SPDEvaluator:
    def __init__(self):
        self.indexer = MultimodalIndexer(force_recreate=False)
        self.retriever = MultimodalRetriever(self.indexer)
        self.generator = MultimodalGenerator()

        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ================= METRICS =================
    def compute_metrics(self, gt, pred):
        gt = gt.lower().strip()
        pred = pred.lower().strip()

        exact = int(gt == pred)
        fuzzy = fuzz.token_set_ratio(gt, pred)

        emb1 = self.sim_model.encode(gt, convert_to_tensor=True)
        emb2 = self.sim_model.encode(pred, convert_to_tensor=True)
        semantic = float(util.cos_sim(emb1, emb2))

        answer_found = int(
            exact == 1 or
            fuzzy > 65 or
            semantic > 0.65
        )

        return exact, fuzzy, semantic, answer_found

    # ================= EVALUATION =================
    def evaluate(
        self,
        excel_path="/content/drive/MyDrive/Question_samples.xlsx",
        output_path="/content/drive/MyDrive/evaluation_results/SPD_Evaluation.xlsx"
    ):
        df = pd.read_excel(excel_path)
        df.columns = ["question", "ground_truth"]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 🔥 Resume logic
        if output_path.exists():
            results_df = pd.read_excel(output_path)
            completed_ids = set(results_df["id"].tolist())
            print(f"Resuming: {len(completed_ids)} already done")
        else:
            results_df = pd.DataFrame()
            completed_ids = set()

        for idx, row in df.iterrows():

            if idx in completed_ids:
                continue  # skip completed

            query = str(row["question"]).strip()
            ground_truth = str(row["ground_truth"]).strip()

            print(f"\n[{idx+1}/{len(df)}] {query[:80]}...")

            start_time = time.time()

            # -------- RETRIEVE --------
            hits = self.retriever.search(
                query_text=query,
                top_k=12,
                source_filter="data/spd.pdf",
            )

            # -------- GENERATE --------
            # gen_output = self.generator.generate_answer(query, hits[:3])

            # answer = gen_output.get("answer", "")
            # usage = gen_output.get("usage", {})
            answer = self.generator.generate_answer(query, hits[:3])

            usage = self.generator.last_usage

            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")

            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")

            latency = time.time() - start_time

            # -------- METRICS --------
            exact, fuzzy, semantic, found = self.compute_metrics(
                ground_truth, answer
            )

            new_row = {
                "id": idx,
                "question": query,
                "ground_truth": ground_truth,
                "generated_answer": answer,

                "exact_match": exact,
                "fuzzy_score": round(fuzzy, 2),
                "semantic_score": round(semantic, 4),
                "answer_found": found,

                "latency_seconds": round(latency, 2),

                # 🔥 TOKEN INFO (REAL, NOT ESTIMATED)
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,

                "retrieved_pages": ", ".join(
                    [f"Page {p.payload.get('page_number')}" for p in hits[:5]]
                ),

                "timestamp": datetime.now().isoformat()
            }

            # 🔥 Append + SAVE IMMEDIATELY
            results_df = pd.concat(
                [results_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            results_df.to_excel(output_path, index=False)

            print(f" Saved → {output_path}")

        # -------- SUMMARY --------
        print("\n===== FINAL SUMMARY =====")
        print(f"Accuracy: {results_df['answer_found'].mean()*100:.2f}%")
        print(f"Avg Latency: {results_df['latency_seconds'].mean():.2f}s")
        print(f"Avg Tokens: {results_df['total_tokens'].mean():.0f}")

        return results_df


if __name__ == "__main__":
    evaluator = SPDEvaluator()
    evaluator.evaluate()