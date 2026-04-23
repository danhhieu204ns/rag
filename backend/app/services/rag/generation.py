from __future__ import annotations

import json

from langchain_core.documents import Document

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


def generate_answer(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> str:
    """Generate answer from question, retrieval context, and chat history."""

    with _timed_query_step(
        "load_chat_llm",
        event_prefix="generate_answer_load_llm",
    ):
        llm = get_llm()

    with _timed_query_step(
        "build_history_block",
        event_prefix="generate_answer_history",
        details={"history_count": len(history_messages)},
    ):
        history_block = "\n".join(
            f"{message.role.upper()}: {message.content}" for message in history_messages[-8:]
        )

    with _timed_query_step(
        "build_context_block",
        event_prefix="generate_answer_context",
        details={"context_doc_count": len(context_docs)},
    ):
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

        context_block = "\n\n".join(context_lines)

        if not context_block:
            context_block = "No retrieved context available."

    prompt = (
        "Bạn là một chuyên gia khoa học tự nhiên thân thiện. "
        "Nhiệm vụ của bạn là giải thích khoa học một cách chính xác, dễ hiểu và gần gũi.\n"
        "Sử dụng đại từ 'mình' và 'bạn' để tạo cảm giác thân thiện như người bạn đồng học.\n"
        "Ưu tiên sử dụng thông tin trong tài liệu được cung cấp. "
        "Nếu tài liệu không đủ căn cứ, hãy nói rõ 'mình chưa tìm thấy đủ thông tin trong tài liệu về điều này' "
        "và tuyệt đối không tự ý thêm kiến thức ngoài nếu mâu thuẫn với tài liệu.\n\n"
        f"--- LỊCH SỬ TRÒ CHUYỆN ---\n{history_block or 'Chưa có tin nhắn trước đó.'}\n\n"
        f"--- TÀI LIỆU HỖ TRỢ ---\n{context_block}\n\n"
        f"--- CÂU HỎI ---\n{question}\n\n"
        "--- YÊU CẦU TRẢ LỜI ---\n"
        "Đánh giá xem câu hỏi là một lời chào/giao tiếp thông thường (chit-chat) hay là một câu hỏi tra cứu kiến thức.\n"
        "1. Nếu là giao tiếp thông thường: Hãy trả lời tự nhiên, thân thiện và KHÔNG sử dụng quy trình 4 bước.\n"
        "2. Nếu là câu hỏi tra cứu: Hãy suy luận theo 4 bước sau. BẮT BUỘC đặt Bước 1, Bước 2, và Bước 3 vào bên trong một khối thẻ <think> và </think>. Bước 4 đặt bên ngoài khối thẻ đó.\n"
        "  <think>\n"
        "  Bước 1 – Phân tích câu hỏi: Xác định các từ khóa chuyên môn và dữ liệu đầu vào quan trọng.\n"
        "  Bước 2 – Truy xuất căn cứ: Liệt kê các định luật, khái niệm hoặc sự kiện khoa học có trong tài liệu liên quan đến câu hỏi.\n"
        "  Bước 3 – Suy luận logic: Kết nối dữ liệu và định luật theo trình tự nguyên nhân – kết quả. Nếu tài liệu không đủ thông tin, ghi rõ 'không đủ căn cứ'.\n"
        "  </think>\n"
        "  Bước 4 – Kết luận cuối cùng: Đưa ra câu trả lời ngắn gọn, dễ hiểu và thân thiện dựa trên suy luận (không lặp lại chữ 'Bước 4').\n\n"
        "Trả lời:"
    )

    with _timed_query_step(
        "invoke_chat_llm",
        event_prefix="generate_answer_invoke_llm",
        details={
            "question_preview": _preview_text(question),
            "context_doc_count": len(context_docs),
        },
    ):
        response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


from typing import Iterator

def generate_answer_stream(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> Iterator[str]:
    """Generate answer from question, retrieval context, and chat history as a stream."""

    with _timed_query_step(
        "load_chat_llm",
        event_prefix="generate_answer_load_llm",
    ):
        llm = get_llm()

    with _timed_query_step(
        "build_history_block",
        event_prefix="generate_answer_history",
        details={"history_count": len(history_messages)},
    ):
        history_block = "\n".join(
            f"{message.role.upper()}: {message.content}" for message in history_messages[-8:]
        )

    with _timed_query_step(
        "build_context_block",
        event_prefix="generate_answer_context",
        details={"context_doc_count": len(context_docs)},
    ):
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

        context_block = "\n\n".join(context_lines)

        if not context_block:
            context_block = "No retrieved context available."

    prompt = (
        "Bạn là một chuyên gia khoa học tự nhiên thân thiện. "
        "Nhiệm vụ của bạn là giải thích khoa học một cách chính xác, dễ hiểu và gần gũi.\n"
        "Sử dụng đại từ 'mình' và 'bạn' để tạo cảm giác thân thiện như người bạn đồng học.\n"
        "Ưu tiên sử dụng thông tin trong tài liệu được cung cấp. "
        "Nếu tài liệu không đủ căn cứ, hãy nói rõ 'mình chưa tìm thấy đủ thông tin trong tài liệu về điều này' "
        "và tuyệt đối không tự ý thêm kiến thức ngoài nếu mâu thuẫn với tài liệu.\n\n"
        f"--- LỊCH SỬ TRÒ CHUYỆN ---\n{history_block or 'Chưa có tin nhắn trước đó.'}\n\n"
        f"--- TÀI LIỆU HỖ TRỢ ---\n{context_block}\n\n"
        f"--- CÂU HỎI ---\n{question}\n\n"
        "--- YÊU CẦU TRẢ LỜI ---\n"
        "Đánh giá xem câu hỏi là một lời chào/giao tiếp thông thường (chit-chat) hay là một câu hỏi tra cứu kiến thức.\n"
        "1. Nếu là giao tiếp thông thường: Hãy trả lời tự nhiên, thân thiện và KHÔNG sử dụng quy trình 4 bước.\n"
        "2. Nếu là câu hỏi tra cứu: Hãy suy luận theo 4 bước sau. BẮT BUỘC đặt Bước 1, Bước 2, và Bước 3 vào bên trong một khối thẻ <think> và </think>. Bước 4 đặt bên ngoài khối thẻ đó.\n"
        "  <think>\n"
        "  Bước 1 – Phân tích câu hỏi: Xác định các từ khóa chuyên môn và dữ liệu đầu vào quan trọng.\n"
        "  Bước 2 – Truy xuất căn cứ: Liệt kê các định luật, khái niệm hoặc sự kiện khoa học có trong tài liệu liên quan đến câu hỏi.\n"
        "  Bước 3 – Suy luận logic: Kết nối dữ liệu và định luật theo trình tự nguyên nhân – kết quả. Nếu tài liệu không đủ thông tin, ghi rõ 'không đủ căn cứ'.\n"
        "  </think>\n"
        "  Bước 4 – Kết luận cuối cùng: Đưa ra câu trả lời ngắn gọn, dễ hiểu và thân thiện dựa trên suy luận (không lặp lại chữ 'Bước 4').\n\n"
        "Trả lời:"
    )

    _emit_query_progress(
        "[chat.query] Start stream response: context_doc_count=%d", len(context_docs)
    )
    for chunk in llm.stream(prompt):
        if hasattr(chunk, "content"):
            yield str(chunk.content)
        else:
            yield str(chunk)



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
