from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException

from ..core.settings import settings


def _headers() -> dict[str, str]:
    if not settings.ollama_api_key:
        return {}
    return {"x-api-key": settings.ollama_api_key}


class SimpleLLMClient:
    """Minimal LLM client for query rewriting and orchestration."""
    
    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}
    
    def invoke(self, prompt: str, temperature: float = 0.0) -> Any:
        """Call LLM with given prompt."""
        timeout = httpx.Timeout(settings.request_timeout_seconds, connect=10.0)
        with httpx.Client(timeout=timeout, headers=self.headers) as client:
            try:
                response = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                    },
                )
            except httpx.TimeoutException as exc:
                raise HTTPException(status_code=504, detail="LLM Service timeout.") from exc
            except httpx.RequestError as exc:
                raise HTTPException(status_code=502, detail=f"Cannot reach LLM Service: {exc}") from exc
        
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise HTTPException(status_code=502, detail="LLM returned no choices.")
        
        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")
        
        class LLMResponse:
            pass
        
        result = LLMResponse()
        result.content = content
        return result


def get_llm_client() -> SimpleLLMClient | None:
    """Get LLM client for query rewriting, or None if not configured."""
    try:
        base_url = settings.ollama_service_url.strip()
        model = getattr(settings, "ollama_chat_model", "mistral")
        api_key = getattr(settings, "ollama_api_key", None)
        
        if not base_url:
            return None
        
        return SimpleLLMClient(base_url, model, api_key)
    except Exception:
        return None


def _extract_embeddings(payload: dict[str, Any]) -> list[list[float]]:
    raw_embeddings = payload.get("embeddings")
    if isinstance(raw_embeddings, list):
        return [
            [float(value) for value in item]
            for item in raw_embeddings
            if isinstance(item, list)
        ]

    raw_embedding = payload.get("embedding")
    if isinstance(raw_embedding, list):
        return [[float(value) for value in raw_embedding]]

    raise HTTPException(status_code=502, detail="Ollama Service did not return embeddings.")


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    timeout = httpx.Timeout(settings.request_timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, headers=_headers()) as client:
        try:
            response = await client.post(
                f"{settings.ollama_service_url}/v1/embed",
                json={"input": texts},
            )
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="Ollama Service embedding timeout.") from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Cannot connect to Ollama Service: {exc}",
            ) from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Ollama Service returned non-JSON response.") from exc

    embeddings = _extract_embeddings(payload)
    if len(embeddings) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=f"Ollama Service returned {len(embeddings)} embeddings for {len(texts)} texts.",
        )
    return embeddings
