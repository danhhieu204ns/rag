from __future__ import annotations

import json
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from ...models import ChatMessage
from .logging import _timed_query_step, _emit_query_progress
from .models import get_llm
from .utils import _to_int, _to_float, _preview_text


def build_sources(context_docs: list[Document]) -> list[dict[str, int | str | float | dict[str, object] | None]]:
    """Extract compact source payload from retrieved chunks."""

    sources: list[dict[str, int | str | float | dict[str, object] | None]] = []
    for doc in context_docs:
        text = doc.page_content.strip().replace("\n", " ")

        source_metadata = doc.metadata.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = None

        sources.append(
            {
                "document_id": int(doc.metadata.get("document_id")) if doc.metadata.get("document_id") is not None else None,
                "chunk_id": _to_int(doc.metadata.get("chunk_id")),
                "chunk_index": _to_int(doc.metadata.get("chunk_index")),
                "page": _to_int(doc.metadata.get("source_page")),
                "source_kind": str(doc.metadata.get("source_kind")) if doc.metadata.get("source_kind") is not None else None,
                "source_metadata": source_metadata,
                "retrieval_mode": str(doc.metadata.get("retrieval_mode")) if doc.metadata.get("retrieval_mode") is not None else None,
                "retrieval_score": _to_float(doc.metadata.get("retrieval_score")),
                "excerpt": text[:280],
            }
        )
    return sources


def _build_context_block(context_docs: list[Document]) -> str:
    context_lines: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        source_metadata = doc.metadata.get("source_metadata")
        source_info = source_metadata.get("source_info") if isinstance(source_metadata, dict) else None
        context = source_metadata.get("context") if isinstance(source_metadata, dict) else None
        search_optimization = source_metadata.get("search_optimization") if isinstance(source_metadata, dict) else None

        meta_parts: list[str] = []
        if isinstance(source_info, dict):
            if source_info.get("file_name"):
                meta_parts.append(f"file={source_info.get('file_name')}")
            if source_info.get("page_number"):
                meta_parts.append(f"page={source_info.get('page_number')}")
            if source_info.get("doc_type"):
                meta_parts.append(f"doc_type={source_info.get('doc_type')}")

        if isinstance(context, dict):
            if context.get("h2"):
                meta_parts.append(f"h2={context.get('h2')}")
            if context.get("h3"):
                meta_parts.append(f"h3={context.get('h3')}")

        if isinstance(search_optimization, dict):
            document_codes = search_optimization.get("document_codes")
            if isinstance(document_codes, list) and document_codes:
                meta_parts.append(f"document_codes={', '.join(str(item) for item in document_codes[:3])}")

        retrieval_mode = doc.metadata.get("retrieval_mode")
        if retrieval_mode is not None:
            meta_parts.append(f"retrieval={retrieval_mode}")

        prefix = f"[Chunk {index}]"
        if meta_parts:
            prefix += " " + " | ".join(meta_parts)

        context_lines.append(prefix)
        context_lines.append(doc.page_content)

    return "\n\n".join(context_lines) or "Không có tài liệu hỗ trợ."


def _build_messages(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> list:
    """Build chat messages for RAG generation."""

    history_block = "\n".join(
        f"{message.role.upper()}: {message.content}" for message in history_messages[-8:]
    )
    context_block = _build_context_block(context_docs)

    system_content = (
        "Bạn là ViettelRAG - trợ lý tra cứu tài liệu chuyên nghiệp của Viettel.\n\n"

        "QUY TẮC TRẢ LỜI:\n\n"

        "1. TRÍCH DẪN NGUỒN (BẮT BUỘC):\n"
        "   Sau mỗi câu có thông tin từ tài liệu, chèn [Chunk N] ngay sau câu đó.\n"
        "   Ví dụ đúng: Viettel thành lập năm 1989 [Chunk 3]. Doanh thu đạt 50 tỷ USD [Chunk 7].\n"
        "   Nếu không có thông tin cụ thể từ tài liệu cho một ý, bỏ qua trích dẫn — KHÔNG viết bất kỳ nội dung ngoặc vuông nào.\n\n"

        "2. ĐÚNG TRỌNG TÂM:\n"
        "   Chỉ trả lời về đúng điều được hỏi.\n"
        "   - Hỏi về mục cụ thể (thứ nhất, loại X...) → chỉ nói về mục đó.\n"
        "   - Hỏi liệt kê/tổng hợp → mới liệt kê đầy đủ.\n\n"

        "3. CHI TIẾT: Khai thác đầy đủ thông tin (định nghĩa, giải thích, ví dụ, số liệu).\n\n"

        "4. ĐỊNH DẠNG: Dùng ### tiêu đề, **in đậm** từ khóa, danh sách -.\n"
        "   Đi thẳng vào nội dung, không thêm tiêu đề dẫn nhập thừa.\n"
        "   Dùng 'mình' và 'bạn'."
    )

    user_content = (
        "=== TÀI LIỆU THAM KHẢO ===\n"
        f"{context_block}\n\n"
        "=== LỊCH SỬ TRÒ CHUYỆN ===\n"
        f"{history_block or 'Chưa có.'}\n\n"
        "=== CÂU HỎI ===\n"
        f"{question}\n\n"
        "Phân tích và trả lời theo định dạng sau:\n"
        "<think>\n"
        "Câu hỏi hỏi về: [xác định đúng đối tượng]\n"
        "Chunks liên quan nhất: [liệt kê số chunk có thông tin trực tiếp]\n"
        "Kế hoạch: [nêu ngắn gọn sẽ trình bày gì]\n"
        "</think>\n\n"
        "[Câu trả lời đầy đủ với [Chunk N] sau mỗi thông tin]"
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]


def generate_answer(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> str:
    """Generate answer from question, retrieval context, and chat history."""

    with _timed_query_step("load_chat_llm", event_prefix="generate_answer_load_llm"):
        llm = get_llm()

    with _timed_query_step(
        "build_prompt",
        event_prefix="generate_answer_build_prompt",
        details={"history_count": len(history_messages), "context_doc_count": len(context_docs)},
    ):
        messages = _build_messages(question, context_docs, history_messages)

    with _timed_query_step(
        "invoke_chat_llm",
        event_prefix="generate_answer_invoke_llm",
        details={
            "question_preview": _preview_text(question),
            "context_doc_count": len(context_docs),
        },
    ):
        response = llm.invoke(messages)

    return str(response.content) if hasattr(response, "content") else str(response)


def generate_answer_stream(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> Iterator[str]:
    """Generate answer as a stream."""

    with _timed_query_step("load_chat_llm", event_prefix="generate_answer_load_llm"):
        llm = get_llm()

    with _timed_query_step(
        "build_prompt",
        event_prefix="generate_answer_build_prompt",
        details={"history_count": len(history_messages), "context_doc_count": len(context_docs)},
    ):
        messages = _build_messages(question, context_docs, history_messages)

    _emit_query_progress(
        "[chat.query] Start stream response: context_doc_count=%d", len(context_docs)
    )

    for chunk in llm.stream(messages):
        content = str(chunk.content) if hasattr(chunk, "content") else str(chunk)
        if content:
            yield content


def parse_sources(raw_json: str | None) -> list[dict[str, int | str | float | dict[str, object] | None]]:
    """Parse serialized sources from chat message payload."""

    if not raw_json:
        return []
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return []
