from __future__ import annotations

import logging
from typing import Any

from ..core.settings import settings
from ..schemas import ContextItem

logger = logging.getLogger(__name__)

_reranker: Any = None
_reranker_failed = False
_reranker_error: str | None = None


def _load_reranker() -> Any:
    global _reranker, _reranker_failed, _reranker_error
    if _reranker is not None or _reranker_failed:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
        logger.info("[reranker] loaded CrossEncoder model=%s", settings.reranker_model)
    except ImportError:
        _reranker_failed = True
        _reranker_error = "sentence-transformers not installed"
        logger.warning("[reranker] %s; returning RRF order", _reranker_error)
    except Exception as exc:
        _reranker_failed = True
        _reranker_error = str(exc)
        logger.exception("[reranker] failed to load model=%s", settings.reranker_model)
    return _reranker


def rerank_contexts(
    *,
    query: str,
    contexts: list[ContextItem],
    top_k: int,
    enabled: bool,
) -> tuple[list[ContextItem], dict[str, Any]]:
    debug: dict[str, Any] = {
        "enabled": enabled,
        "model": settings.reranker_model,
        "input_count": len(contexts),
        "output_count": min(top_k, len(contexts)),
        "status": "disabled" if not enabled else "pending",
    }
    if not enabled:
        return contexts[:top_k], debug
    if len(contexts) <= top_k:
        debug["status"] = "skipped_pool_size"
        return contexts[:top_k], debug

    reranker = _load_reranker()
    if reranker is None:
        debug["status"] = "unavailable"
        debug["error"] = _reranker_error
        return contexts[:top_k], debug

    pairs = [(query, item.content) for item in contexts]
    scores = reranker.predict(pairs)
    scored = [
        (context, float(score))
        for context, score in zip(contexts, scores)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    reranked: list[ContextItem] = []
    for item, score in scored[:top_k]:
        metadata = dict(item.metadata or {})
        metadata["reranker_score"] = score
        reranked.append(
            ContextItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                content=item.content,
                source=item.source,
                page=item.page,
                score=item.score,
                metadata=metadata,
            )
        )

    debug.update(
        {
            "status": "ok",
            "output_count": len(reranked),
            "original_top_score": contexts[0].score if contexts else None,
            "reranked_top_score": scored[0][1] if scored else None,
            "score_preview": [
                {
                    "chunk_id": item.chunk_id,
                    "reranker_score": score,
                    "rrf_score": item.score,
                }
                for item, score in scored[: min(top_k, 10)]
            ],
        }
    )
    return reranked, debug
