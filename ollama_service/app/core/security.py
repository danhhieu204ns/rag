from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from .settings import settings


logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
_rate_bucket: dict[str, dict[str, Any]] = {}
_rate_lock = threading.Lock()


def verify_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    if not settings.shield_api_key:
        logger.error("[ollama-service] SHIELD_API_KEY not configured")
        raise HTTPException(status_code=500, detail="Server chưa cấu hình SHIELD_API_KEY.")

    if not api_key:
        logger.warning("[ollama-service][security] missing x-api-key header")
        raise HTTPException(status_code=401, detail="Thiếu header x-api-key.")

    if not secrets.compare_digest(api_key, settings.shield_api_key):
        logger.warning("[ollama-service][security] invalid api key attempt")
        raise HTTPException(status_code=403, detail="API key không hợp lệ.")

    logger.debug("[ollama-service][security] api_key verified")
    return api_key


def enforce_rate_limit(api_key: str, route_name: str) -> None:
    now = int(time.time())
    window = now // 60
    bucket_key = f"{api_key}:{route_name}:{window}"

    with _rate_lock:
        item = _rate_bucket.get(bucket_key, {"count": 0, "window": window})
        item["count"] += 1
        _rate_bucket[bucket_key] = item

        for key, value in list(_rate_bucket.items()):
            if value["window"] < window:
                _rate_bucket.pop(key, None)

        count = int(item["count"])

    logger.debug("[ollama-service][ratelimit] route=%s count=%d/%d window=%d", route_name, count, settings.rate_limit_per_minute, window)

    if count > settings.rate_limit_per_minute:
        logger.warning("[ollama-service][ratelimit] limit exceeded route=%s count=%d limit=%d", route_name, count, settings.rate_limit_per_minute)
        raise HTTPException(
            status_code=429,
            detail=f"Vượt giới hạn {settings.rate_limit_per_minute} request/phút cho endpoint {route_name}.",
        )
