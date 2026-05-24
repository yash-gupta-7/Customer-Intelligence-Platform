"""
auth.py — API key authentication for protected endpoints.
"""
from typing import Optional
from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate X-API-Key header against settings.API_SECRET_KEY."""
    settings = get_settings()
    if not api_key or api_key != settings.api_secret_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
