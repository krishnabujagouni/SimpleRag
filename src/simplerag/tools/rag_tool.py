


# import os, hashlib
# from typing import List, Dict
# import lancedb
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from crewai.tools import tool


# # --- Config ---
# PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")
# MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

# # --- Hash ID helper
# def _make_id(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()

# # --- Universal Embedding Wrapper
# def embed_text(text: str) -> List[float]:
#     """Embed text with user-selected provider."""
#     if PROVIDER == "gemini":
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

# def embed_query(query: str) -> List[float]:
#     """Embed query with user-selected provider."""
#     if PROVIDER == "gemini":
#         import google.generativeai as genai
#         try:
#             api_key = os.getenv("GOOGLE_API_KEY")
#             genai.configure(api_key=api_key)
            
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
#             return resp["embedding"]
#         except Exception as e:
#             print(f"Error calling Gemini API: {e}")
#             raise
#     else:
#         return embed_text(query)  # Fallback

# # --- Simple chunker
# def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# # --- Ingest Function
# @tool
# def rag_ingest(document: str) -> str:
#     """Split, embed, and store document in LanceDB."""
#     try:
#         chunks = chunk_text(document)
#         embeddings = [embed_text(c) for c in chunks]
        
#         # Create DataFrame with proper structure
#         data = []
#         for chunk, emb in zip(chunks, embeddings):
#             data.append({
#                 "id": _make_id(chunk),
#                 "text": chunk,
#                 "embedding": emb  # This should be a list of floats
#             })
        
#         df = pd.DataFrame(data)
        
#         # Connect to LanceDB
#         db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
        
#         # Create or append to table
#         if "documents" in db.table_names():
#             # Append to existing table
#             tbl = db.open_table("documents")
#             tbl.add(df)
#         else:
#             # Create new table from DataFrame
#             tbl = db.create_table("documents", df)
        
#         return f"✅ Ingested {len(data)} chunks with {len(embeddings[0])}-dim embeddings using {PROVIDER}/{MODEL}."
#     except Exception as e:
#         return f"❌ Error ingesting document: {str(e)}"

# @tool
# # --- Retrieve Function with fallback to manual similarity search
# def rag_retrieve(query: str, top_k: int = 3) -> List[Dict]:
#     """Retrieve top-k relevant chunks for query."""
#     try:
#         # Connect to database
#         db = lancedb.connect(os.getenv("VECTOR_DB_PATH", "./.lancedb"))
        
#         if "documents" not in db.table_names():
#             return [{"error": "No documents have been ingested yet. Please ingest some documents first."}]
        
#         tbl = db.open_table("documents")
        
#         # Get all data as DataFrame for manual processing
#         df = tbl.to_pandas()
#         if len(df) == 0:
#             return [{"error": "No documents found in the database. Please ingest some documents first."}]
        
#         print(f"Found {len(df)} documents in database")
        
#         # Get query embedding
#         query_embedding = embed_query(query)
#         print(f"Query embedding dimension: {len(query_embedding)}")
        
#         # Try LanceDB vector search first
#         try:
#             # Method 1: Try with vector_column_name parameter
#             results = tbl.search(query_embedding, vector_column_name="embedding").limit(top_k).to_list()
            
#             # Format LanceDB results
#             formatted_results = []
#             for i, result in enumerate(results):
#                 formatted_results.append({
#                     "rank": i + 1,
#                     "text": result.get("text", ""),
#                     "id": result.get("id", ""),
#                     "distance": result.get("_distance", 0.0)
#                 })
#             return formatted_results
            
#         except Exception as lance_error:
#             print(f"LanceDB search failed: {lance_error}")
#             print("Falling back to manual similarity search...")
            
#             # Fallback Method: Manual cosine similarity
#             try:
#                 # Extract embeddings from DataFrame
#                 doc_embeddings = []
#                 doc_texts = []
#                 doc_ids = []
                
#                 for _, row in df.iterrows():
#                     doc_embeddings.append(row['embedding'])
#                     doc_texts.append(row['text'])
#                     doc_ids.append(row['id'])
                
#                 # Calculate cosine similarities
#                 query_vec = np.array(query_embedding).reshape(1, -1)
#                 doc_vecs = np.array(doc_embeddings)
                
#                 similarities = cosine_similarity(query_vec, doc_vecs)[0]
                
#                 # Get top-k most similar documents
#                 top_indices = np.argsort(similarities)[::-1][:top_k]
                
#                 formatted_results = []
#                 for i, idx in enumerate(top_indices):
#                     formatted_results.append({
#                         "rank": i + 1,
#                         "text": doc_texts[idx],
#                         "id": doc_ids[idx],
#                         "similarity": float(similarities[idx])
#                     })
                
#                 return formatted_results
                
#             except Exception as manual_error:
#                 return [{"error": f"Both LanceDB search and manual search failed. LanceDB: {lance_error}, Manual: {manual_error}"}]
        
#     except Exception as e:
#         import traceback
#         print(f"Full error traceback: {traceback.format_exc()}")
#         return [{"error": f"Error retrieving documents: {str(e)}"}]


# def ingest_text(text: str, api_key: str) -> str:
#     """
#     Text-based ingestion entrypoint for routes.
#     """
#     try:
#         return rag_ingest(text)
#     except Exception as e:
#         return f"❌ Failed to ingest text: {e}"


# def ingest(file_path: str, api_key: str) -> str:
#     """
#     File-based ingestion entrypoint for routes.
#     Reads file -> passes to rag_ingest().
#     """
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             text = f.read()
#         return rag_ingest(text)
#     except Exception as e:
#         return f"❌ Failed to ingest {file_path}: {e}"


# def retrieve(query: str, api_key: str) -> List[Dict]:
#     """
#     Query-based retrieval entrypoint for routes.
#     Simply forwards to rag_retrieve().
#     """
#     return rag_retrieve(query)




import os, hashlib
from typing import List, Dict
import lancedb
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from crewai.tools import tool


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

# --- Core Functions (without @tool decoration) ---
def _rag_ingest_function(document: str) -> str:
    """Core ingest logic without @tool decoration"""
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

def _rag_retrieve_function(query: str, top_k: int = 3) -> List[Dict]:
    """Core retrieve logic without @tool decoration"""
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

# --- Tool Decorated Functions (for CrewAI) ---
@tool
def rag_ingest(document: str) -> str:
    """Split, embed, and store document in LanceDB."""
    return _rag_ingest_function(document)

@tool
def rag_retrieve(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve top-k relevant chunks for query."""
    return _rag_retrieve_function(query, top_k)

# --- Route Helper Functions (call core functions directly) ---
def ingest_text(text: str, api_key: str) -> str:
    """
    Text-based ingestion entrypoint for routes.
    Calls the core function directly, not the tool wrapper.
    """
    try:
        return _rag_ingest_function(text)
    except Exception as e:
        return f"❌ Failed to ingest text: {e}"

def ingest(file_path: str, api_key: str) -> str:
    """
    File-based ingestion entrypoint for routes.
    Reads file -> passes to core ingest function.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return _rag_ingest_function(text)
    except Exception as e:
        return f"❌ Failed to ingest {file_path}: {e}"

def retrieve(query: str, api_key: str) -> List[Dict]:
    """
    Query-based retrieval entrypoint for routes.
    Calls the core function directly, not the tool wrapper.
    """
    try:
        return _rag_retrieve_function(query)
    except Exception as e:
        return [{"error": f"❌ Failed to retrieve: {e}"}]