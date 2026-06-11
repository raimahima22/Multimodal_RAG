from src.agent import run_agent

if __name__ == "__main__":
    test_queries = [
        "What is the deductible for this plan?",
        "What does the SPD say about pre-existing conditions?",
        "Compare out-of-pocket maximum between SBC and SPD",
    ]
    
    for q in test_queries:
        print(f"\n🔹 Query: {q}")
        print(f"Answer: {run_agent(q)}")
        print("-" * 80)