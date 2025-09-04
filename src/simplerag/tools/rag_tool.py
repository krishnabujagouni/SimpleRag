# from crewai.tools import tool
# import os, hashlib
# from typing import List, Dict
# import lancedb
# # Import pyarrow for explicit schema definition
# import pyarrow as pa

# # --- Config ---
# PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")
# MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# # --- Hash ID helper
# def _make_id(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()

# # --- Universal Embedding Wrapper
# def embed_text(text: str, task_type: str) -> List[float]:
#     """
#     Embed text with user-selected provider.
#     Includes task_type for Gemini optimization.
#     """
#     if PROVIDER == "openai":
#         from openai import OpenAI
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         resp = client.embeddings.create(model=MODEL, input=text)
#         return resp.data[0].embedding

#     elif PROVIDER == "ollama":
#         import ollama
#         resp = ollama.embeddings(model=MODEL, prompt=text)
#         return resp["embedding"]

#     elif PROVIDER == "gemini":
#         import google.generativeai as genai
#         try:
#             api_key = os.getenv("GOOGLE_API_KEY")
#             if not api_key:
#                 raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
            
#             genai.configure(api_key=api_key)
            
#             # Use the newer model name format and pass the task_type
#             resp = genai.embed_content(
#                 model=MODEL if MODEL.startswith("models/") else f"models/{MODEL}",
#                 content=text,
#                 task_type=task_type
#             )
#             return resp["embedding"]
#         except Exception as e:
#             print(f"Error calling Gemini API: {e}")
#             raise
#     else:
#         raise ValueError(f"Unsupported provider: {PROVIDER}")

# # --- LanceDB init with EXPLICIT SCHEMA ---
# def get_db(embedding_dim: int):
#     """
#     Connects to LanceDB and creates the table with an explicit PyArrow schema
#     to prevent the 'no vector column' error.
#     """
#     db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))

#     schema = pa.schema([
#         pa.field("id", pa.string()),
#         pa.field("text", pa.string()),
#         # This explicitly tells LanceDB that 'embedding' is the vector column
#         pa.field("embedding", lancedb.schema.vector(embedding_dim))
#     ])

#     if "documents" not in db.table_names():
#         db.create_table("documents", schema=schema)
#     return db

# # --- Simple chunker ---
# def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# # --- Ingest Tool ---
# @tool("rag_ingest")
# def rag_ingest(document: str) -> str:
#     """Split, embed, and store document in LanceDB."""
#     try:
#         chunks = chunk_text(document)
#         if not chunks:
#             return "No content to ingest."
        
#         # Embed all chunks at once for document-type embedding
#         embeddings = [embed_text(c, task_type="RETRIEVAL_DOCUMENT") for c in chunks]
#         dim = len(embeddings[0])
        
#         db = get_db(dim)
#         tbl = db.open_table("documents")
        
#         data = [{"id": _make_id(chunk), "text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
#         tbl.add(data)
        
#         return f"✅ Ingested {len(data)} chunks with {dim}-dim embeddings."
#     except Exception as e:
#         return f"❌ Error ingesting document: {str(e)}"

# # --- Retrieve Tool ---
# @tool("rag_retrieve")
# def rag_retrieve(query: str, top_k: int = 3) -> List[Dict]:
#     """Retrieve top-k relevant chunks for a query."""
#     try:
#         db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
#         if "documents" not in db.table_names():
#             return [{"error": "No documents have been ingested yet."}]
        
#         tbl = db.open_table("documents")
#         if len(tbl) == 0:
#             return [{"error": "The document database is empty."}]

#         # Embed the query with the specific 'retrieval_query' task type
#         emb = embed_text(query, task_type="RETRIEVAL_QUERY")
        
#         results = tbl.search(emb).limit(top_k).to_list()
        
#         # Format results nicely
#         return [
#             {
#                 "rank": i + 1,
#                 "text": result.get("text", ""),
#                 "id": result.get("id", ""),
#                 "score": result.get("_distance", "N/A")
#             }
#             for i, result in enumerate(results)
#         ]
#     except Exception as e:
#         return [{"error": f"Error retrieving documents: {str(e)}"}]


# working code
# import os, hashlib
# from typing import List, Dict
# import lancedb
# from lancedb.pydantic import LanceModel
# from typing import Annotated
# from lancedb.pydantic import Vector

# # --- Config ---
# PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")
# MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# # --- Hash ID helper
# def _make_id(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()

# # --- Universal Embedding Wrapper
# def embed_text(text: str) -> List[float]:
#     """Embed text with user-selected provider."""
#     if PROVIDER == "openai":
#         from openai import OpenAI
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         resp = client.embeddings.create(model=MODEL, input=text)
#         return resp.data[0].embedding
    
#     elif PROVIDER == "ollama":
#         import ollama
#         resp = ollama.embeddings(model=MODEL, prompt=text)
#         return resp["embedding"]
    
#     elif PROVIDER == "gemini":
#         import google.generativeai as genai
#         try:
#             api_key = os.getenv("GOOGLE_API_KEY")
#             if not api_key:
#                 raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
            
#             genai.configure(api_key=api_key)
            
#             # For text-embedding-004, use the newer API format
#             if MODEL == "text-embedding-004":
#                 resp = genai.embed_content(
#                     model=f"models/{MODEL}",
#                     content=text,
#                     task_type="retrieval_document"  # Optimize for retrieval
#                 )
#             else:
#                 # For older models like embedding-001
#                 resp = genai.embed_content(
#                     model=MODEL if MODEL.startswith("models/") else f"models/{MODEL}",
#                     content=text,
#                     task_type="retrieval_document"
#                 )
#             return resp["embedding"]
#         except Exception as e:
#             print(f"Error calling Gemini API: {e}")
#             raise
    
#     else:
#         raise ValueError(f"Unsupported provider: {PROVIDER}")

# # --- LanceDB init (dynamic schema)
# def get_db(embedding_dim: int):
#     db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
    
#     class DocumentChunk(LanceModel):
#         id: str
#         text: str
#         embedding: Annotated[List[float], Vector(embedding_dim)]
    
#     if "documents" not in db.table_names():
#         db.create_table("documents", schema=DocumentChunk)
#     return db

# # --- Simple chunker (can replace with token-based later)
# def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# # --- Ingest Function (NO @tool decorator)
# def rag_ingest(document: str) -> str:
#     """Split, embed, and store document in LanceDB."""
#     try:
#         chunks = chunk_text(document)
#         embeddings = [embed_text(c) for c in chunks]
#         dim = len(embeddings[0])
        
#         db = get_db(dim)
#         tbl = db.open_table("documents")
        
#         data = []
#         for chunk, emb in zip(chunks, embeddings):
#             data.append({
#                 "id": _make_id(chunk),
#                 "text": chunk,
#                 "embedding": emb
#             })
#         tbl.add(data)
        
#         return f"✅ Ingested {len(data)} chunks with {dim}-dim embeddings using {PROVIDER}/{MODEL}."
#     except Exception as e:
#         return f"❌ Error ingesting document: {str(e)}"

# # --- Retrieve Function (NO @tool decorator)
# def rag_retrieve(query: str, top_k: int = 3) -> List[Dict]:
#     """Retrieve top-k relevant chunks for query."""
#     try:
#         # Use retrieval_query task type for queries
#         if PROVIDER == "gemini":
#             import google.generativeai as genai
#             genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            
#             # For text-embedding-004, use the newer API format
#             if MODEL == "text-embedding-004":
#                 resp = genai.embed_content(
#                     model=f"models/{MODEL}",
#                     content=query,
#                     task_type="retrieval_query"  # Optimize for queries
#                 )
#             else:
#                 resp = genai.embed_content(
#                     model=MODEL if MODEL.startswith("models/") else f"models/{MODEL}",
#                     content=query,
#                     task_type="retrieval_query"
#                 )
#             emb = resp["embedding"]
#         else:
#             emb = embed_text(query)
        
#         dim = len(emb)
#         db = get_db(dim)
#         tbl = db.open_table("documents")
        
#         results = tbl.search(emb).limit(top_k).to_list()
#         return results
#     except Exception as e:
#         return [{"error": f"Error retrieving documents: {str(e)}"}]

# working for ingestion not reterviewing
# import os, hashlib
# from typing import List, Dict
# import lancedb
# from lancedb.pydantic import LanceModel
# from typing import Annotated
# from lancedb.pydantic import Vector

# # --- Config ---
# PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")
# MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# # --- Hash ID helper
# def _make_id(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()

# # --- Universal Embedding Wrapper
# def embed_text(text: str) -> List[float]:
#     """Embed text with user-selected provider."""
#     if PROVIDER == "openai":
#         from openai import OpenAI
#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         resp = client.embeddings.create(model=MODEL, input=text)
#         return resp.data[0].embedding
    
#     elif PROVIDER == "ollama":
#         import ollama
#         resp = ollama.embeddings(model=MODEL, prompt=text)
#         return resp["embedding"]
    
#     elif PROVIDER == "gemini":
#         import google.generativeai as genai
#         try:
#             api_key = os.getenv("GOOGLE_API_KEY")
#             if not api_key:
#                 raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
            
#             genai.configure(api_key=api_key)
            
#             # For text-embedding-004, use the newer API format
#             if MODEL == "text-embedding-004":
#                 resp = genai.embed_content(
#                     model=f"models/{MODEL}",
#                     content=text,
#                     task_type="retrieval_document"  # Optimize for retrieval
#                 )
#             else:
#                 # For older models like embedding-001
#                 resp = genai.embed_content(
#                     model=MODEL if MODEL.startswith("models/") else f"models/{MODEL}",
#                     content=text,
#                     task_type="retrieval_document"
#                 )
#             return resp["embedding"]
#         except Exception as e:
#             print(f"Error calling Gemini API: {e}")
#             raise
    
#     else:
#         raise ValueError(f"Unsupported provider: {PROVIDER}")

# # --- LanceDB init (dynamic schema)
# def get_db(embedding_dim: int):
#     db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
    
#     class DocumentChunk(LanceModel):
#         id: str
#         text: str
#         embedding: Annotated[List[float], Vector(embedding_dim)]
    
#     if "documents" not in db.table_names():
#         db.create_table("documents", schema=DocumentChunk)
#     return db

# # --- Simple chunker (can replace with token-based later)
# def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# # --- Ingest Function (NO @tool decorator)
# def rag_ingest(document: str) -> str:
#     """Split, embed, and store document in LanceDB."""
#     try:
#         chunks = chunk_text(document)
#         embeddings = [embed_text(c) for c in chunks]
#         dim = len(embeddings[0])
        
#         db = get_db(dim)
#         tbl = db.open_table("documents")
        
#         data = []
#         for chunk, emb in zip(chunks, embeddings):
#             data.append({
#                 "id": _make_id(chunk),
#                 "text": chunk,
#                 "embedding": emb
#             })
#         tbl.add(data)
        
#         return f"✅ Ingested {len(data)} chunks with {dim}-dim embeddings using {PROVIDER}/{MODEL}."
#     except Exception as e:
#         return f"❌ Error ingesting document: {str(e)}"

# # --- Retrieve Function (NO @tool decorator)
# def rag_retrieve(query: str, top_k: int = 3) -> List[Dict]:
#     """Retrieve top-k relevant chunks for query."""
#     try:
#         # First check if database and table exist
#         db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
        
#         if "documents" not in db.table_names():
#             return [{"error": "No documents have been ingested yet. Please ingest some documents first."}]
        
#         tbl = db.open_table("documents")
        
#         # Check if table has any data
#         try:
#             row_count = len(tbl.to_pandas())
#             if row_count == 0:
#                 return [{"error": "No documents found in the database. Please ingest some documents first."}]
#             print(f"Found {row_count} documents in database")
#         except Exception as e:
#             return [{"error": f"Error checking table contents: {str(e)}"}]
        
#         # Get query embedding
#         if PROVIDER == "gemini":
#             import google.generativeai as genai
#             genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            
#             # For text-embedding-004, use the newer API format
#             if MODEL == "text-embedding-004":
#                 resp = genai.embed_content(
#                     model=f"models/{MODEL}",
#                     content=query,
#                     task_type="retrieval_query"  # Optimize for queries
#                 )
#             else:
#                 resp = genai.embed_content(
#                     model=MODEL if MODEL.startswith("models/") else f"models/{MODEL}",
#                     content=query,
#                     task_type="retrieval_query"
#                 )
#             emb = resp["embedding"]
#         else:
#             emb = embed_text(query)
        
#         print(f"Query embedding dimension: {len(emb)}")
        
#         # Perform vector search
#         results = tbl.search(emb).limit(top_k).to_list()
        
#         # Format results nicely
#         formatted_results = []
#         for i, result in enumerate(results):
#             formatted_results.append({
#                 "rank": i + 1,
#                 "text": result.get("text", ""),
#                 "id": result.get("id", ""),
#                 "score": result.get("_distance", "N/A")  # LanceDB returns _distance
#             })
        
#         return formatted_results
        
#     except Exception as e:
#         return [{"error": f"Error retrieving documents: {str(e)}"}]
    



import os, hashlib
from typing import List, Dict
import lancedb
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Config ---
PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")
MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# --- Hash ID helper
def _make_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# --- Universal Embedding Wrapper
def embed_text(text: str) -> List[float]:
    """Embed text with user-selected provider."""
    if PROVIDER == "gemini":
        import google.generativeai as genai
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
            
            genai.configure(api_key=api_key)
            
            # For text-embedding-004, use the newer API format
            if MODEL == "text-embedding-004":
                resp = genai.embed_content(
                    model=f"models/{MODEL}",
                    content=text,
                    task_type="retrieval_document"  # Optimize for retrieval
                )
            else:
                # For older models like embedding-001
                resp = genai.embed_content(
                    model=MODEL if MODEL.startswith("models/") else f"models/{MODEL}",
                    content=text,
                    task_type="retrieval_document"
                )
            return resp["embedding"]
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")

def embed_query(query: str) -> List[float]:
    """Embed query with user-selected provider."""
    if PROVIDER == "gemini":
        import google.generativeai as genai
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            
            # For text-embedding-004, use the newer API format
            if MODEL == "text-embedding-004":
                resp = genai.embed_content(
                    model=f"models/{MODEL}",
                    content=query,
                    task_type="retrieval_query"  # Optimize for queries
                )
            else:
                resp = genai.embed_content(
                    model=MODEL if MODEL.startswith("models/") else f"models/{MODEL}",
                    content=query,
                    task_type="retrieval_query"
                )
            return resp["embedding"]
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise
    else:
        return embed_text(query)  # Fallback

# --- Simple chunker
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# --- Ingest Function
def rag_ingest(document: str) -> str:
    """Split, embed, and store document in LanceDB."""
    try:
        chunks = chunk_text(document)
        embeddings = [embed_text(c) for c in chunks]
        
        # Create DataFrame with proper structure
        data = []
        for chunk, emb in zip(chunks, embeddings):
            data.append({
                "id": _make_id(chunk),
                "text": chunk,
                "embedding": emb  # This should be a list of floats
            })
        
        df = pd.DataFrame(data)
        
        # Connect to LanceDB
        db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
        
        # Create or append to table
        if "documents" in db.table_names():
            # Append to existing table
            tbl = db.open_table("documents")
            tbl.add(df)
        else:
            # Create new table from DataFrame
            tbl = db.create_table("documents", df)
        
        return f"✅ Ingested {len(data)} chunks with {len(embeddings[0])}-dim embeddings using {PROVIDER}/{MODEL}."
    except Exception as e:
        return f"❌ Error ingesting document: {str(e)}"

# --- Retrieve Function with fallback to manual similarity search
def rag_retrieve(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve top-k relevant chunks for query."""
    try:
        # Connect to database
        db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
        
        if "documents" not in db.table_names():
            return [{"error": "No documents have been ingested yet. Please ingest some documents first."}]
        
        tbl = db.open_table("documents")
        
        # Get all data as DataFrame for manual processing
        df = tbl.to_pandas()
        if len(df) == 0:
            return [{"error": "No documents found in the database. Please ingest some documents first."}]
        
        print(f"Found {len(df)} documents in database")
        
        # Get query embedding
        query_embedding = embed_query(query)
        print(f"Query embedding dimension: {len(query_embedding)}")
        
        # Try LanceDB vector search first
        try:
            # Method 1: Try with vector_column_name parameter
            results = tbl.search(query_embedding, vector_column_name="embedding").limit(top_k).to_list()
            
            # Format LanceDB results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "text": result.get("text", ""),
                    "id": result.get("id", ""),
                    "distance": result.get("_distance", 0.0)
                })
            return formatted_results
            
        except Exception as lance_error:
            print(f"LanceDB search failed: {lance_error}")
            print("Falling back to manual similarity search...")
            
            # Fallback Method: Manual cosine similarity
            try:
                # Extract embeddings from DataFrame
                doc_embeddings = []
                doc_texts = []
                doc_ids = []
                
                for _, row in df.iterrows():
                    doc_embeddings.append(row['embedding'])
                    doc_texts.append(row['text'])
                    doc_ids.append(row['id'])
                
                # Calculate cosine similarities
                query_vec = np.array(query_embedding).reshape(1, -1)
                doc_vecs = np.array(doc_embeddings)
                
                similarities = cosine_similarity(query_vec, doc_vecs)[0]
                
                # Get top-k most similar documents
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                formatted_results = []
                for i, idx in enumerate(top_indices):
                    formatted_results.append({
                        "rank": i + 1,
                        "text": doc_texts[idx],
                        "id": doc_ids[idx],
                        "similarity": float(similarities[idx])
                    })
                
                return formatted_results
                
            except Exception as manual_error:
                return [{"error": f"Both LanceDB search and manual search failed. LanceDB: {lance_error}, Manual: {manual_error}"}]
        
    except Exception as e:
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return [{"error": f"Error retrieving documents: {str(e)}"}]