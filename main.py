# main.py

from src.indexing import index_documents
from llm_agent import answer_query

def main():
    print("Indexing docs...")
    index_documents()

    print("Enter query:")
    q = input("> ")
    ans = answer_query(q)
    print("\nLLM Answer:")
    print(ans)

if __name__ == "__main__":
    main()