# chat.py
import os
import mlflow
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# Load environment variables
load_dotenv()

# LangSmith Config
LANGSMITH_ENABLED = os.getenv("LANGSMITH_API_KEY") is not None and os.getenv("LANGSMITH_API_KEY") != ""

# MLflow Config
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
MLFLOW_ENABLED = False
try:
    mlflow.set_experiment("LLMops_Multi_Doc_Chat")
    MLFLOW_ENABLED = True
except Exception as e:
    print(f"Warning: Could not connect to MLflow server: {e}. Tracking will be disabled.")

# Config
COLLECTION_NAME = "primmod_paper"
QDRANT_PATH = "./vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
client = QdrantClient(path=QDRANT_PATH)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# LLM choice: OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

retriever = vector_store.as_retriever(search_kwargs={"k": 20})

prompt_template = """
You are a knowledgeable research assistant. Answer the user's question based on the provided context.
Try to be comprehensive and structured in your answer. 
If the information is not present in the context, explicitly state what is missing or if you can't find it.

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# RAG Chain
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def _get_answer_logic(question: str):
    """Core logic for getting an answer"""
    if not MLFLOW_ENABLED:
        try:
            answer = rag_chain.invoke(question)
            docs = retriever.invoke(question)
            sources = list(set([doc.metadata.get('source') for doc in docs if doc.metadata.get('source')]))
            return {"answer": answer, "sources": sources}
        except Exception as e:
            return {"error": str(e)}

    with mlflow.start_run(nested=True):
        try:
            mlflow.log_param("question", question)
            answer = rag_chain.invoke(question)
            
            # Get sources for metadata
            docs = retriever.invoke(question)
            sources = list(set([doc.metadata.get('source') for doc in docs if doc.metadata.get('source')]))
            
            mlflow.log_param("answer", answer)
            mlflow.log_metric("source_count", len(sources))
            
            return {"answer": answer, "sources": sources}
        except Exception as e:
            mlflow.log_error(str(e))
            return {"error": str(e)}

def get_answer(question: str):
    """Public function that optionally uses LangSmith tracing"""
    if LANGSMITH_ENABLED:
        try:
            return traceable(_get_answer_logic)(question)
        except Exception as e:
            # If LangSmith fails (e.g., 403), fall back to normal execution
            if "Forbidden" in str(e) or "403" in str(e):
                print("Warning: LangSmith tracing failed (possibly invalid key). Running without tracing.")
            return _get_answer_logic(question)
    return _get_answer_logic(question)

def chat_loop():
    print("\n=== Multi-Document PDF Chat (OpenAI) ===\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        if not question:
            continue
        print("Thinking...")
        result = get_answer(question)
        if "error" in result:
            print(f"Error: {result['error']}\n")
        else:
            print(f"AI: {result['answer']}\n")
            print("Sources used:", result['sources'])

if __name__ == "__main__":
    chat_loop()