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


def _build_generation_prompt(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> str:
    history_block = "\n".join(
        f"{message.role.upper()}: {message.content}" for message in history_messages[-8:]
    )
    context_block = _build_context_block(context_docs)

    return (
        "Bạn là \"Chuyên gia Phân tích Tài liệu Hệ thống RAG\". Nhiệm vụ của bạn là cung cấp "
        "câu trả lời chuyên sâu, chi tiết và có căn cứ xác thực TUYỆT ĐỐI từ tài liệu được cung cấp.\n\n"
        
        "--- NGUYÊN TẮC HÀNH VI ---\n"
        "1. TÍNH CHÍNH XÁC: Chỉ trả lời dựa trên dữ liệu có trong tài liệu. Nếu tài liệu không nói, "
        "hãy khẳng định là không có thông tin. Tuyệt đối không suy diễn ngoài văn bản.\n"
        "2. ĐỘ CHI TIẾT: Khi giải thích một khái niệm hoặc trả lời câu hỏi, hãy trình bày đầy đủ "
        "các khía cạnh liên quan có trong tài liệu (nguyên nhân, diễn biến, kết quả, các con số thống kê...).\n"
        "3. XỬ LÝ DANH SÁCH: Hiểu các ký hiệu đánh số (1, 2, 3...) hoặc ký tự (a, b, c...) là thứ tự "
        "ưu tiên hoặc thứ tự xuất hiện (ví dụ: \"1.\" tương ứng với \"Thứ nhất\").\n"
        "4. TRÍCH DẪN: BẮT BUỘC chèn thẻ [Chunk ID] ngay sau mỗi thông tin cụ thể được trích lục.\n\n"
        
        "--- CẤU TRÚC PHẢN HỒI ---\n"
        "Mọi phản hồi phải tuân thủ cấu trúc sau:\n\n"
        "<think>\n"
        "- Bước 1: Xác định các thực thể chính và mục tiêu tìm kiếm trong câu hỏi.\n"
        "- Bước 2: Lọc ra các Chunk chứa thông tin trực tiếp. Loại bỏ các Chunk nhiễu (chỉ chứa từ khóa nhưng không chứa nội dung trả lời).\n"
        "- Bước 3: Nếu câu hỏi yêu cầu giải thích chi tiết, hãy liên kết dữ liệu từ nhiều Chunk để xây dựng một bức tranh toàn cảnh.\n"
        "- Bước 4: Kiểm chứng lại: \"Thông tin này có thực sự nằm trong văn bản không?\"\n"
        "</think>\n\n"
        
        "--- CÂU TRẢ LỜI (Trình bày chuyên nghiệp) ---\n"
        "Sử dụng đại từ \"mình\" và \"bạn\". Trình bày theo phong cách phân tích chuyên sâu:\n"
        "- Nếu là câu hỏi tra cứu: Đi thẳng vào nội dung chính, sau đó mở rộng bằng các chi tiết bổ trợ "
        "có trong tài liệu để làm rõ vấn đề.\n"
        "- Nếu có danh sách: Hãy liệt kê đầy đủ và giải thích từng mục dựa trên nội dung tài liệu.\n"
        "- Định dạng: Sử dụng các thẻ tiêu đề (###), in đậm (**) các từ khóa quan trọng để dễ theo dõi.\n\n"
        
        "--- DỮ LIỆU ĐẦU VÀO ---\n"
        f"LỊCH SỬ TRÒ CHUYỆN:\n{history_block or 'Trống.'}\n\n"
        f"TÀI LIỆU HỖ TRỢ:\n{context_block}\n\n"
        f"CÂU HỎI CỦA NGƯỜI DÙNG: {question}\n\n"
        
        "Trả lời:"
    )


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
        prompt = _build_generation_prompt(question, context_docs, history_messages)

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

    with _timed_query_step("load_chat_llm", event_prefix="generate_answer_load_llm"):
        llm = get_llm()

    with _timed_query_step(
        "build_prompt",
        event_prefix="generate_answer_build_prompt",
        details={"history_count": len(history_messages), "context_doc_count": len(context_docs)},
    ):
        prompt = _build_generation_prompt(question, context_docs, history_messages)

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
