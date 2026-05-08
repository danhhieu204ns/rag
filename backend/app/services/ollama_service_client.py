from __future__ import annotations

from ..core.settings import settings


def require_ollama_service_url() -> str:
    service_url = settings.ollama_base_url.strip()
    if not service_url:
        raise RuntimeError(
            "OLLAMA_BASE_URL is not configured. Backend requires the ollama_service for chat, embedding, and indexing."
        )
    return service_url


def require_ollama_api_key() -> str:
    api_key = settings.ollama_api_key.strip()
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not configured. Backend requires the ollama_service API key."
        )
    return api_key


def ollama_service_headers() -> dict[str, str]:
    return {"x-api-key": require_ollama_api_key()}
