import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from chat import rag_chain, retriever
from dotenv import load_dotenv

# Load env vars
load_dotenv()

def run_ragas_evaluation():
    print("Starting Ragas Evaluation...")
    
    # Define test dataset
    # In a real scenario, you'd load this from a file
    test_data = [
        {
            "question": "What is the main topic of the PrimMod paper?",
            "ground_truth": "The PrimMod paper discusses a primitive-based modeling approach for AI workshop paper constraints and automation."
        },
        {
            "question": "How does PrimMod handle AI workshop paper constraints?",
            "ground_truth": "PrimMod uses a modular approach to decompose constraints into manageable primitives and then reconstructs them for final verification."
        }
    ]
    
    questions = [item["question"] for item in test_data]
    ground_truths = [item["ground_truth"] for item in test_data]
    answers = []
    contexts = []
    
    print("Generating answers and retrieving contexts...")
    for q in questions:
        # Get answer
        response = rag_chain.invoke(q)
        answers.append(response)
        
        # Get contexts
        docs = retriever.invoke(q)
        contexts.append([doc.page_content for doc in docs])
    
    # Create dataset for Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)
    
    # Run evaluation
    print("Running Ragas metrics...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    # Save results
    result_df = result.to_pandas()
    result_df.to_csv("evaluation_results.csv", index=False)
    
    with open("evaluation_results.json", "w") as f:
        json.dump(result, f, indent=4)
        
    print("\nEvaluation Results:")
    print(result)
    print("\nResults saved to evaluation_results.csv and evaluation_results.json")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment.")
    else:
        run_ragas_evaluation()
