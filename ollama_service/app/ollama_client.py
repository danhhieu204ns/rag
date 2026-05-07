from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import HTTPException
from pydantic import BaseModel

from .core.settings import settings


def model_dump(obj: BaseModel) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return obj.dict(exclude_none=True)


def validate_text_length(text: str, max_chars: int, field_name: str) -> None:
    if len(str(text or "")) > max_chars:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} quá dài. Tối đa {max_chars} ký tự.",
        )


def validate_text_batch(
    texts: str | list[str],
    *,
    max_items: int,
    max_chars: int,
    field_name: str,
) -> None:
    if isinstance(texts, str):
        validate_text_length(texts, max_chars, field_name)
        return

    if len(texts) > max_items:
        raise HTTPException(status_code=413, detail=f"Tối đa {max_items} đoạn text/lần.")

    for index, item in enumerate(texts):
        validate_text_length(str(item), max_chars, f"{field_name}[{index}]")


def enforce_num_predict(payload: dict[str, Any], max_num_predict: int) -> dict[str, Any]:
    payload["stream"] = False
    options = dict(payload.get("options") or {})

    try:
        requested = int(options.get("num_predict", max_num_predict))
    except (TypeError, ValueError):
        requested = max_num_predict

    options["num_predict"] = min(requested, max_num_predict)
    payload["options"] = options
    return payload


def _timeout(total_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(
        total_seconds,
        connect=settings.ollama_connect_timeout_seconds,
    )


async def post_ollama(path: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_timeout(timeout_seconds)) as client:
        try:
            response = await client.post(f"{settings.ollama_base_url}{path}", json=payload)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"Ollama timeout khi gọi {path}.") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Không kết nối được Ollama: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama trả response không phải JSON từ {path}.") from exc


async def get_ollama(path: str, *, timeout_seconds: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_timeout(timeout_seconds)) as client:
        try:
            response = await client.get(f"{settings.ollama_base_url}{path}")
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"Ollama timeout khi gọi {path}.") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Không kết nối được Ollama: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama trả response không phải JSON từ {path}.") from exc


def parse_ollama_json_response(ollama_result: dict[str, Any]) -> dict[str, Any]:
    raw_response = str(ollama_result.get("response") or "")
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Model không trả về JSON hợp lệ.",
                "raw_response": raw_response,
            },
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Model JSON response không phải object.",
                "raw_response": raw_response,
            },
        )
    return parsed


def usage_payload(ollama_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": ollama_result.get("model"),
        "total_duration": ollama_result.get("total_duration"),
        "prompt_eval_count": ollama_result.get("prompt_eval_count"),
        "eval_count": ollama_result.get("eval_count"),
    }
