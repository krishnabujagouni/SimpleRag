from fastapi import Header, HTTPException
from key_manager import validate_api_key

def verify_api_key(x_api_key: str = Header(...)):
    """
    Dependency for FastAPI routes.
    Validates API key provided in request headers.
    """
    if not validate_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key
