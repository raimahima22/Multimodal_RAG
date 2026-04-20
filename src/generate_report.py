# generate_report.py
import pandas as pd
from datetime import datetime
from load_dotenv
df = pd.read_excel("evaluation_results/SPD_Evaluation_xxxxxxxx.xlsx")


report = f"""
# SPD Multimodal RAG - Evaluation Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Document:** SPD Only
**Total Questions Evaluated:** {len(df)}

##  Summary
- Average Latency: **{df['latency_seconds'].mean():.2f} seconds**
- Questions Answered: {len(df)}
- Fastest Answer: {df['latency_seconds'].min():.2f}s
- Slowest Answer: {df['latency_seconds'].max():.2f}s

## Next Steps Recommendation
1. Manually review 10-15 low-quality answers
2. Add `expected_page` column later for better retrieval eval
3. Improve OCR / reranking if many answers are weak

**Detailed results saved in:** SPD_Evaluation_xxxx.xlsx
"""

print(report)
Path("SPD_FINAL_REPORT.md").write_text(report)