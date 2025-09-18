from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load env first
load_dotenv()

app = FastAPI(title="SimpleRAG API")

# Ensure knowledge directory structure exists
def setup_knowledge_structure():
    """Create the knowledge_base directory structure"""
    # FIXED: Use raw strings or os.path.join for Windows paths
    PROJECT_ROOT = r"C:\simplerag"
    knowledge_root = os.path.join(PROJECT_ROOT, "knowledge")
    
    directories = [
        knowledge_root,
        os.path.join(knowledge_root, "raw"),
        os.path.join(knowledge_root, "processed")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Ensured directory exists: {directory}")

@app.on_event("startup")
async def startup_event():
    # Setup directory structure
    setup_knowledge_structure()
    
    # Set the correct vector DB path
    os.environ["VECTOR_DB_PATH"] = "./knowledge/.lancedb"
    
    print("✅ Using provider:", os.getenv("EMBED_PROVIDER"))
    print("✅ Using model:", os.getenv("EMBED_MODEL"))
    print("✅ Google API key loaded:", "Yes" if os.getenv("GOOGLE_API_KEY") else "No")
    print("✅ Vector DB path:", os.getenv("VECTOR_DB_PATH"))
    print("✅ Knowledge base directory structure created")

from routes import router
app.include_router(router, prefix="/api")
# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "service": "RAG-as-a-Service"}


