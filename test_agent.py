# test_agent.py  (place this in the root folder)
from src.agent import run_agent

def test_routing():
    test_cases = [
        ("What is the deductible?", "SBC"),
        ("What are the plan eligibility rules?", "SPD"),
        ("What is the out-of-pocket maximum?", "SBC"),
        ("Explain prior authorization process", "SPD"),
        ("Compare deductible in SBC and SPD", "Both"),
        ("What services are covered under this plan?", "SBC"),
    ]
    
    print("🧪 Testing LangGraph Tool Routing\n")
    
    for query, expected in test_cases:
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        answer = run_agent(query)
        print(f"Answer: {answer[:300]}..." if len(answer) > 300 else f"Answer: {answer}")
        print("-" * 90 + "\n")

if __name__ == "__main__":
    test_routing()