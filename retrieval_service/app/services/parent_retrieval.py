from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChildHit:
    child_chunk_id: str
    parent_id: int | None
    document_id: int | None
    child_text: str
    score: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParentRecord:
    parent_id: int
    document_id: int
    text: str
    page: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParentContext:
    parent_id: int | None
    document_id: int | None
    text: str
    page: int | None
    score: float | None
    metadata: dict[str, Any]


def expand_child_hits_to_parent_contexts(
    child_hits: list[ChildHit],
    parent_records: dict[int, ParentRecord],
    *,
    final_top_k: int,
    deduplicate: bool = True,
) -> list[ParentContext]:
    contexts: list[ParentContext] = []
    seen_parent_ids: set[int] = set()

    for hit in child_hits:
        parent_id = hit.parent_id
        if parent_id is not None and deduplicate:
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)

        parent = parent_records.get(parent_id) if parent_id is not None else None
        metadata = dict(hit.metadata)
        metadata["child_chunk_id"] = hit.child_chunk_id
        metadata["parent_id"] = parent_id
        if hit.score is not None:
            metadata["retrieval_score"] = hit.score

        if parent is None:
            metadata["parent_missing"] = parent_id is not None
            contexts.append(
                ParentContext(
                    parent_id=parent_id,
                    document_id=hit.document_id,
                    text=hit.child_text,
                    page=None,
                    score=hit.score,
                    metadata=metadata,
                )
            )
        else:
            parent_metadata = dict(parent.metadata)
            parent_metadata.update(metadata)
            contexts.append(
                ParentContext(
                    parent_id=parent.parent_id,
                    document_id=parent.document_id,
                    text=parent.text,
                    page=parent.page,
                    score=hit.score,
                    metadata=parent_metadata,
                )
            )

        if len(contexts) >= final_top_k:
            break

    return contexts
