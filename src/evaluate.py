import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime

from src.indexer import MultimodalIndexer
from src.retriever import MultimodalRetriever
from src.generator import MultimodalGenerator


class SPDEvaluator:
    def __init__(self):
        self.indexer = MultimodalIndexer(force_recreate=False)
        self.retriever = MultimodalRetriever(self.indexer)
        self.generator = MultimodalGenerator()
        
    def evaluate(self, excel_path: str = "spd_questions.xlsx", output_dir="evaluation_results"):
        excel_path = "/content/drive/MyDrive/Question_samples.xlsx"
        
        print(f" Reading questions from: {excel_path}")
        
        try:
            df = pd.read_excel(excel_path)
            print(f" Successfully loaded {len(df)} questions from Question_samples.xlsx")
        except FileNotFoundError:
            print(f" File not found: {excel_path}")
            print("Make sure the file is uploaded to MyDrive and Drive is mounted!")
            return None
        except Exception as e:
            print(f" Error reading Excel: {e}")
            return None
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Starting SPD Evaluation on {len(df)} questions...\n")
        
        for idx, row in df.iterrows():
            query = str(row['question']).strip()
            ground_truth = str(row.get('ground_truth_answer', '')).strip()
            
            print(f"[{idx+1}/{len(df)}] {query[:100]}...")

            start_time = time.time()

            # Force SPD only
            hits = self.retriever.search(
                query_text=query,
                top_k=12,
                source_filter="data/spd.pdf",   # Hardcoded for SPD
                generator=self.generator
            )

            answer = self.generator.generate_answer(query, hits[:3])

            latency = time.time() - start_time

            result = {
                "id": idx,
                "question": query,
                "ground_truth": ground_truth,
                "generated_answer": answer,
                "latency_seconds": round(latency, 2),
                "retrieved_pages": [f"Page {p.payload.get('page_number')}" for p in hits[:5]],
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)

            if (idx + 1) % 5 == 0:
                self._save_results(results, output_dir)

        # Final save
        final_df = self._save_results(results, output_dir)
        return final_df

    def _save_results(self, results, output_dir):
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        excel_path = output_dir / f"SPD_Evaluation_{timestamp}.xlsx"
        df_results.to_excel(excel_path, index=False)
        
        print(f"Saved: {excel_path}")
        return df_results


# ========================= RUN IT =========================
if __name__ == "__main__":
    evaluator = SPDEvaluator()
    results_df = evaluator.evaluate("spd_questions.xlsx")   # Change filename if needed