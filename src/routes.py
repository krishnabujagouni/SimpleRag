

import os
from fastapi import APIRouter, UploadFile, File, Depends
from auth import verify_api_key
from key_manager import generate_api_key
from simplerag.tools import rag_tool
from pydantic import BaseModel
from simplerag.crew import RagCrew

# Initialize crews once
rag_ingest_crew = RagCrew().ingest_crew()
rag_retrieve_crew = RagCrew().retrieve_crew()

router = APIRouter()

UPLOAD_DIR = "knowledge_base/raw"

# -------------------
# 🔑 Admin: Create API Key
# -------------------
@router.post("/admin/create-key")
def create_key():
    key = generate_api_key()
    return {"api_key": key}

# -------------------
# 📝 Ingest Raw Text
# -------------------
class IngestTextRequest(BaseModel):
    text: str

@router.post("/ingest_text")
async def ingest_text(req: IngestTextRequest, api_key: str = Depends(verify_api_key)):
    # Ensure API key is set
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Some LiteLLM versions need this
    
    result = rag_tool.ingest_text(req.text, api_key)
    
    # 🚀 Trigger Crew
    crew_result = rag_ingest_crew.kickoff(inputs={"text": req.text})
    
    return {"status": "success", "message": result, "crew_result": crew_result}

# -------------------
# 📂 Upload Document
# -------------------
@router.post("/upload-1")
async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    # Ensure API key is set
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Some LiteLLM versions need this
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    rag_tool.ingest(file_path, api_key)
    
    # 🚀 Trigger Crew (no parentheses - use the already initialized crew)
    crew_result = rag_ingest_crew.kickoff(inputs={"file": file.filename})
    
    return {"status": "success", "filename": file.filename, "crew_result": crew_result}

# -------------------
# 🔍 Query Documents
# -------------------
class QueryRequest(BaseModel):
    query: str

@router.post("/query")
async def query_documents(req: QueryRequest, api_key: str = Depends(verify_api_key)):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    # Use the direct function instead of the tool wrapper
    result = rag_tool.retrieve(req.query, api_key)
    
    # 🚀 Trigger Crew - Fix the extra parentheses
    crew_result = rag_retrieve_crew.kickoff(inputs={"query": req.query})
    
    return {"query": req.query, "results": result, "crew_result": crew_result}