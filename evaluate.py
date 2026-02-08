import mlflow
from chat import get_answer

def run_evaluation():
    test_questions = [
        "What is the main topic of the PrimMod paper?",
        "How does PrimMod handle AI workshop paper constraints?",
        "What are the key results mentioned in the documents?"
    ]
    
    with mlflow.start_run(run_name="Initial RAG Evaluation"):
        for q in test_questions:
            print(f"Testing: {q}")
            result = get_answer(q)
            if "answer" in result:
                print(f"Answer received (length: {len(result['answer'])})")
            else:
                print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    run_evaluation()
