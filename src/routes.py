

# import os
# from fastapi import APIRouter, UploadFile, File, Depends
# from auth import verify_api_key
# from key_manager import generate_api_key
# from simplerag.tools import rag_tool
# from pydantic import BaseModel
# from simplerag.crew import RagCrew

# # Initialize crews once
# rag_crew = RagCrew()
# rag_ingest_crew = RagCrew().ingest_crew()
# rag_retrieve_crew = RagCrew().retrieve_crew()

# router = APIRouter()

# # # UPLOAD_DIR = "/raw"
# # UPLOAD_DIR = "C:\simplerag\knowledge\raw"
# # # PROCESSED_DIR = "knowledge/processed"
# # VECTOR_DB_PATH = "C:\simplerag\knowledge\.lancedb"

# UPLOAD_DIR = r"C:\simplerag\knowledge\raw"
# PROCESSED_DIR = r"C:\simplerag\knowledge\processed"
# VECTOR_DB_PATH = r"C:\simplerag\knowledge\.lancedb"



# # Ensure knowledge_base directories exist
# os.makedirs("C:\simplerag\knowledge", exist_ok=True)
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# # os.makedirs(PROCESSED_DIR, exist_ok=True)

# # -------------------
# # 🔑 Admin: Create API Key
# # -------------------
# @router.post("/admin/create-key")
# def create_key():
#     key = generate_api_key()
#     return {"api_key": key}

# # -------------------
# # 📝 Ingest Raw Text
# # -------------------
# class IngestTextRequest(BaseModel):
#     text: str

# @router.post("/ingest_text")
# async def ingest_text(req: IngestTextRequest, api_key: str = Depends(verify_api_key)):
#     # Ensure API key is set
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#     # os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Some LiteLLM versions need this
#     os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
#     result = rag_tool.ingest_text(req.text, api_key)
    
#     # 🚀 Trigger Crew
#     crew_result = rag_ingest_crew.kickoff(inputs={"text": req.text})
    
#     return {"status": "success", "message": result, "crew_result": crew_result}

# # -------------------
# # 📂 Upload Document
# # -------------------
# @router.post("/upload-1")
# async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
#     # Ensure API key is set
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#     # os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY") # Some LiteLLM versions need this
#     os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
    
#     rag_tool.ingest(file_path, api_key)
    
#     # 🚀 Trigger Crew (no parentheses - use the already initialized crew)
#     crew_result = rag_ingest_crew.kickoff(inputs={"file": file.filename})
    
#     return {"status": "success", "filename": file.filename, "crew_result": crew_result}

# # -------------------
# # 🔍 Query Documents
# # -------------------
# class QueryRequest(BaseModel):
#     query: str

# @router.post("/query")
# async def query_documents(req: QueryRequest, api_key: str = Depends(verify_api_key)):
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#     os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
#     # Use the direct function instead of the tool wrapper
#     result = rag_tool.retrieve(req.query, api_key)
    
#     # 🚀 Trigger Crew - Fix the extra parentheses
#     crew_result = rag_retrieve_crew.kickoff(inputs={"query": req.query})
    
#     return {"query": req.query, "results": result, "crew_result": crew_result}



# import os
# from fastapi import APIRouter, UploadFile, File, Depends
# from auth import verify_api_key
# from key_manager import generate_api_key
# from simplerag.tools import rag_tool
# from pydantic import BaseModel
# from simplerag.crew import RagCrew

# # Initialize crews once (keeping your existing structure)
# # rag_crew = RagCrew()
# rag_ingest_crew = RagCrew.ingest_crew()
# rag_retrieve_crew = RagCrew.retrieve_crew()

# router = APIRouter()

# # Directory paths
# UPLOAD_DIR = r"C:\simplerag\knowledge\raw"
# PROCESSED_DIR = r"C:\simplerag\knowledge\processed"
# VECTOR_DB_PATH = r"C:\simplerag\knowledge\.lancedb"

# # Ensure knowledge_base directories exist
# os.makedirs(r"C:\simplerag\knowledge", exist_ok=True)
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(PROCESSED_DIR, exist_ok=True)

# # -------------------
# # 🔑 Admin: Create API Key
# # -------------------
# @router.post("/admin/create-key")
# def create_key():
#     key = generate_api_key()
#     return {"api_key": key}

# # -------------------
# # 📝 Ingest Raw Text (Fixed input passing)
# # -------------------
# class IngestTextRequest(BaseModel):
#     text: str

# @router.post("/ingest_text")
# async def ingest_text(req: IngestTextRequest, api_key: str = Depends(verify_api_key)):
#     # Set environment variables
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#     os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
#     try:
#         # Option 1: Use direct tool call only (simpler, more reliable)
#         result = rag_tool.ingest_text(req.text, api_key)
        
#         # Option 2: If you want to use crew, pass input with correct key
#         # The task needs to know what to ingest - pass the actual text
#         crew_inputs = {
#             "document": req.text,  # This should match your task's expected input variable
#             "text": req.text       # Alternative key in case task expects 'text'
#         }
        
#         crew_result = rag_ingest_crew.kickoff(inputs=crew_inputs)
        
#         return {
#             "status": "success", 
#             "message": result,
#             "crew_result": str(crew_result),
#             "input_type": "text",
#             "input_length": len(req.text)
#         }
        
#     except Exception as e:
#         # If crew fails, still return the direct tool result
#         try:
#             result = rag_tool.ingest_text(req.text, api_key)
#             return {
#                 "status": "partial_success", 
#                 "message": result,
#                 "crew_error": str(e),
#                 "note": "Direct ingestion successful, crew execution failed"
#             }
#         except Exception as tool_error:
#             return {
#                 "status": "error", 
#                 "message": f"Both crew and direct tool failed. Crew: {str(e)}, Tool: {str(tool_error)}"
#             }

# # -------------------
# # 📂 Upload Document (Works with your existing crew)
# # -------------------
# @router.post("/upload")
# async def upload_document(
#     file: UploadFile = File(...), 
#     api_key: str = Depends(verify_api_key)
# ):
#     # Set environment variables
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#     os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
#     try:
#         os.makedirs(UPLOAD_DIR, exist_ok=True)
#         file_path = os.path.join(UPLOAD_DIR, file.filename)
        
#         # Save uploaded file
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
        
#         # Use direct tool call for file ingestion
#         result = rag_tool.ingest(file_path, api_key)
        
#         # Trigger crew with file input (matches your task config)
#         crew_result = rag_ingest_crew.kickoff(inputs={"file": file.filename, "file_path": file_path})
        
#         return {
#             "status": "success", 
#             "filename": file.filename,
#             "file_path": file_path,
#             "message": result,
#             "crew_result": str(crew_result),
#             "input_type": "file"
#         }
        
#     except Exception as e:
#         return {
#             "status": "error", 
#             "message": f"Failed to upload and process file: {str(e)}"
#         }

# # -------------------
# # 🔍 Query Documents (Works with your existing crew)
# # -------------------
# class QueryRequest(BaseModel):
#     query: str

# @router.post("/query")
# async def query_documents(req: QueryRequest, api_key: str = Depends(verify_api_key)):
#     os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#     os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
#     try:
#         # Use direct tool call for retrieval
#         result = rag_tool.retrieve(req.query, api_key)
        
#         # Trigger crew with query input (matches your task config)
#         crew_result = rag_retrieve_crew.kickoff(inputs={"query": req.query})
        
#         return {
#             "query": req.query,
#             "results": result,
#             "crew_result": str(crew_result),
#             "status": "success"
#         }
        
#     except Exception as e:
#         return {
#             "status": "error",
#             "query": req.query,
#             "message": f"Failed to query documents: {str(e)}"
#         }

# # -------------------
# # 📊 Additional Utility Endpoints
# # -------------------

# @router.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "message": "RAG system is running"}

# @router.get("/supported_formats")
# async def get_supported_formats():
#     """Get list of supported document formats"""
#     try:
#         from simplerag.tools.rag_tool import get_supported_formats
#         formats = get_supported_formats()
#         return {"supported_formats": formats}
#     except Exception as e:
#         return {"error": f"Could not retrieve formats: {str(e)}"}

# @router.get("/database_info")
# async def get_database_info(api_key: str = Depends(verify_api_key)):
#     """Get information about the vector database"""
#     try:
#         import lancedb
#         os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
        
#         db = lancedb.connect(VECTOR_DB_PATH)
#         table_names = db.table_names()
        
#         if "documents" in table_names:
#             tbl = db.open_table("documents")
#             count = tbl.count_rows()
#             return {
#                 "status": "success",
#                 "database_path": VECTOR_DB_PATH,
#                 "tables": table_names,
#                 "document_count": count
#             }
#         else:
#             return {
#                 "status": "success",
#                 "database_path": VECTOR_DB_PATH,
#                 "tables": table_names,
#                 "document_count": 0,
#                 "message": "No documents table found"
#             }
            
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Could not access database: {str(e)}"
#         }



import os
from fastapi import APIRouter, UploadFile, File, Depends
from auth import verify_api_key
from key_manager import generate_api_key
from simplerag.tools import rag_tool
from pydantic import BaseModel
from simplerag.crew import RagCrew

# Initialize crews once - FIXED: Use single instance
rag_crew_instance = RagCrew()  # Create one instance
rag_ingest_crew = rag_crew_instance.ingest_crew()  # Use same instance
rag_retrieve_crew = rag_crew_instance.retrieve_crew()  # Use same instance

router = APIRouter()

UPLOAD_DIR = r"C:\simplerag\knowledge\raw"
PROCESSED_DIR = r"C:\simplerag\knowledge\processed"
VECTOR_DB_PATH = r"C:\simplerag\knowledge\.lancedb"

# Ensure knowledge_base directories exist
os.makedirs(r"C:\simplerag\knowledge", exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
    result = rag_tool.ingest_text(req.text, api_key)
    
    # 🚀 Trigger Crew - FIXED: Use "user_input" to match your task yaml
    crew_result = rag_ingest_crew.kickoff(inputs={"user_input": req.text})
    
    return {"status": "success", "message": result, "crew_result": crew_result}

# -------------------
# 📂 Upload Document
# -------------------
@router.post("/upload-1")
async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    # Ensure API key is set
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    rag_tool.ingest(file_path, api_key)
    
    # 🚀 Trigger Crew - Use file_path as the user_input (agent will detect it's a file)
    crew_result = rag_ingest_crew.kickoff(inputs={"user_input": file_path})
    
    return {"status": "success", "filename": file.filename, "crew_result": crew_result}

# -------------------
# 🔍 Query Documents
# -------------------
class QueryRequest(BaseModel):
    query: str
    file_format: str = ""

@router.post("/query")
async def query_documents(req: QueryRequest, api_key: str = Depends(verify_api_key)):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    
    # Use the direct function instead of the tool wrapper
    result = rag_tool.retrieve(req.query, api_key)
    
    # 🚀 Trigger Crew
    crew_result = rag_retrieve_crew.kickoff(inputs={"query": req.query, "file_format": req.file_format})
    
    return {"query": req.query, "results": result, "crew_result": crew_result}