

# from fastapi import FastAPI, UploadFile, File, HTTPException
# import os
# import shutil
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from simplerag.crew import RagCrew 

# # Load environment variables from .env file
# load_dotenv()

# # ---------- Setup ----------
# app = FastAPI(title="RAG-as-a-Service", version="1.0")
# UPLOAD_DIR = "./data/uploads/"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Instantiate the main RagCrew class
# rag_crew_setup = RagCrew()

# # Create two separate, dedicated crews at startup
# ingest_crew = rag_crew_setup.ingest_crew()
# retrieve_crew = rag_crew_setup.retrieve_crew()


# # ---------- Pydantic Models for API Body ----------
# class IngestTextRequest(BaseModel):
#     text: str

# class RetrieveRequest(BaseModel):
#     query: str
#     top_k: int = 3


# # ---------- API Endpoints ----------
# @app.post("/ingest_text")
# async def ingest_text(req: IngestTextRequest):
#     """
#     Accepts raw text and kicks off the ingestion crew.
#     """
#     try:
#         result = ingest_crew.kickoff(inputs={"document": req.text})
#         return {"status": "success", "message": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     """
#     Accepts a file upload and kicks off the ingestion crew.
#     """
#     try:
#         file_path = os.path.join(UPLOAD_DIR, file.filename)
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#             document_text = f.read()

#         result = ingest_crew.kickoff(inputs={"document": document_text})
#         return {"status": "success", "message": f"File '{file.filename}' ingested.", "crew_result": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/retrieve")
# async def retrieve_data(req: RetrieveRequest):
#     """
#     Accepts a query and kicks off the retrieval crew.
#     """
#     try:
#         result = retrieve_crew.kickoff(inputs={"query": req.query, "top_k": req.top_k})
#         return {"query": req.query, "results": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load env first
load_dotenv()

app = FastAPI(title="SimpleRAG API")

@app.on_event("startup")
async def startup_event():
    print("✅ Using provider:", os.getenv("EMBED_PROVIDER"))
    print("✅ Using model:", os.getenv("EMBED_MODEL"))
    print("✅ Google API key loaded:", "Yes" if os.getenv("GOOGLE_API_KEY") else "No")

from routes import router
app.include_router(router, prefix="/api")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "service": "RAG-as-a-Service"}


