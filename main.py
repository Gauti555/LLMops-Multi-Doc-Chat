import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from mangum import Mangum
from chat import get_answer, client, embeddings
from ingest import ingest_pdf
import shutil

app = FastAPI(title="LLMops Multi-Doc Chat API")

# Mangum wrapper for AWS Lambda
handler = Mangum(app)

class ChatRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    pdf_path: str = None  # Optional path, defaults to set value

@app.get("/")
async def root():
    return {"message": "Welcome to the LLMops Multi-Doc Chat API"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    result = get_answer(request.question)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest = None):
    # Triggers ingestion for a path already on the server
    pdf_path = request.pdf_path if request and request.pdf_path else None
    result = ingest_pdf(pdf_path, client=client, embeddings=embeddings) if pdf_path else ingest_pdf(client=client, embeddings=embeddings)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.post("/upload-pdf")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Ingest the uploaded file with shared resources
    result = ingest_pdf(file_path, client=client, embeddings=embeddings)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"message": f"Successfully uploaded and ingested {file.filename}", "details": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
