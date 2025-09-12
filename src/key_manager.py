import secrets
import json
import os

KEYS_FILE = "keys.json"

def load_keys():
    if not os.path.exists(KEYS_FILE):
        return []
    with open(KEYS_FILE, "r") as f:
        return json.load(f)

def save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def generate_api_key() -> str:
    """Generate a secure random API key"""
    key = secrets.token_hex(16)  # 32-char hex string
    keys = load_keys()
    if key not in keys:
        keys.append(key)
        save_keys(keys)
    return key

def validate_api_key(key: str) -> bool:
    keys = load_keys()
    return key in keys
