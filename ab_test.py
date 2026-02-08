import os
import mlflow
from chat import rag_chain, retriever, llm
from langchain_openai import ChatOpenAI

def run_ab_test():
    print("Starting A/B Test...")
    
    test_questions = [
        "What is PrimMod?",
        "What are the benefits of using primitives in modeling?"
    ]
    
    configs = [
        {"name": "GPT-4o-Mini-Low-Temp", "model": "gpt-4o-mini", "temp": 0.1},
        {"name": "GPT-4o-Mini-High-Temp", "model": "gpt-4o-mini", "temp": 0.9}
    ]
    
    for config in configs:
        with mlflow.start_run(run_name=f"AB_Test_{config['name']}"):
            mlflow.log_params({
                "model": config["model"],
                "temperature": config["temp"]
            })
            
            # Setup LLM with this config
            test_llm = ChatOpenAI(model=config["model"], temperature=config["temp"])
            
            # Reconstruct chain with test LLM
            # (In a more robust setup, you'd pass the LLM to get_answer)
            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            
            prompt_template = """Answer based on context:\n{context}\n\nQuestion: {question}"""
            prompt = PromptTemplate.from_template(prompt_template)
            
            test_chain = (
                {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
                 "question": RunnablePassthrough()}
                | prompt
                | test_llm
                | StrOutputParser()
            )
            
            for q in test_questions:
                print(f"Testing Config [{config['name']}] with Question: {q}")
                answer = test_chain.invoke(q)
                mlflow.log_text(answer, f"answer_{q[:20].replace(' ', '_')}.txt")
                
    print("A/B test completed. Check MLflow UI for results.")

if __name__ == "__main__":
    run_ab_test()
