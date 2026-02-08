import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "data/Final Version PrimMod4AI___workshop _Paper.pdf"
VECTOR_STORE_PATH = "./vector_store"
COLLECTION_NAME = "primmod_paper"

def ingest_pdf(pdf_path=PDF_PATH, client=None, embeddings=None):
    if not os.path.exists(pdf_path):
        return {"error": f"File not found at {pdf_path}"}

    print(f"Loading PDF from: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if client is None:
        client = QdrantClient(path=VECTOR_STORE_PATH)
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME in collection_names:
            client.delete_collection(COLLECTION_NAME)
            
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384, 
                distance=models.Distance.COSINE
            )
        )

        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
        vector_store.add_documents(texts)
        
        return {"status": "success", "chunks": len(texts)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    result = ingest_pdf()
    print(result)