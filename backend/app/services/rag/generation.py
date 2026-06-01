from __future__ import annotations

import json
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from ...models import ChatMessage
from .logging import _timed_query_step, _emit_query_progress
from .models import get_llm
from .utils import _to_int, _to_float, _preview_text


# ─── Source utilities ─────────────────────────────────────────────────────────

def build_sources(
    context_docs: list[Document],
) -> list[dict[str, int | str | float | dict | None]]:
    """Extract compact source payload from retrieved chunks."""
    sources = []
    for doc in context_docs:
        text = doc.page_content.strip().replace("\n", " ")
        source_metadata = doc.metadata.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = None
        sources.append({
            "document_id": int(doc.metadata["document_id"])
                if doc.metadata.get("document_id") is not None else None,
            "chunk_id": _to_int(doc.metadata.get("chunk_id")),
            "chunk_index": _to_int(doc.metadata.get("chunk_index")),
            "page": _to_int(doc.metadata.get("source_page")),
            "source_kind": str(doc.metadata["source_kind"])
                if doc.metadata.get("source_kind") is not None else None,
            "source_metadata": source_metadata,
            "retrieval_mode": str(doc.metadata["retrieval_mode"])
                if doc.metadata.get("retrieval_mode") is not None else None,
            "retrieval_score": _to_float(doc.metadata.get("retrieval_score")),
            "excerpt": text[:280],
        })
    return sources


def _build_context_block(context_docs: list[Document]) -> str:
    lines: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        source_metadata = doc.metadata.get("source_metadata")
        source_info = source_metadata.get("source_info") if isinstance(source_metadata, dict) else None
        context = source_metadata.get("context") if isinstance(source_metadata, dict) else None
        search_opt = source_metadata.get("search_optimization") if isinstance(source_metadata, dict) else None

        meta: list[str] = []
        if isinstance(source_info, dict):
            if source_info.get("file_name"):
                meta.append(f"file={source_info['file_name']}")
            if source_info.get("page_number"):
                meta.append(f"page={source_info['page_number']}")
            if source_info.get("doc_type"):
                meta.append(f"doc_type={source_info['doc_type']}")
        if isinstance(context, dict):
            if context.get("h2"):
                meta.append(f"h2={context['h2']}")
            if context.get("h3"):
                meta.append(f"h3={context['h3']}")
        if isinstance(search_opt, dict):
            codes = search_opt.get("document_codes")
            if isinstance(codes, list) and codes:
                meta.append(f"document_codes={', '.join(str(c) for c in codes[:3])}")
        if doc.metadata.get("retrieval_mode"):
            meta.append(f"retrieval={doc.metadata['retrieval_mode']}")

        header = f"--- Chunk {index}"
        if meta:
            header += " | " + " | ".join(meta)
        header += " ---"
        lines.append(header)
        lines.append(doc.page_content)

    return "\n\n".join(lines) or "Không có tài liệu hỗ trợ."


# ─── System prompts ───────────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = """\
Bạn là Trợ lý tri thức VTAca - trợ lý tra cứu tài liệu chuyên nghiệp của Viettel Academy.
Tài liệu tham khảo được đánh số theo ký hiệu "--- Chunk N ---". Khi trích dẫn trong câu trả lời, dùng [Chunk N].

QUY TẮC TRẢ LỜI:

1. TRÍCH DẪN NGUỒN (BẮT BUỘC):
   [Chunk N] PHẢI đặt ở CUỐI câu, ngay trước dấu chấm câu. TUYỆT ĐỐI không đặt [Chunk N] ở đầu câu.
   ✅ ĐÚNG: Viettel thành lập năm 1989 [Chunk 3]. Doanh thu đạt 50 tỷ USD [Chunk 7].
   ❌ SAI:  [Chunk 3] Viettel thành lập năm 1989.
   ❌ SAI:  [Chunk 3] Trên đây là... ← KHÔNG BAO GIỜ bắt đầu câu bằng [Chunk N].
   Nếu không có thông tin cụ thể từ tài liệu cho một ý, bỏ qua trích dẫn.

2. ĐÚNG TRỌNG TÂM:
   Chỉ trả lời về đúng điều được hỏi.
   - Hỏi về mục cụ thể (thứ nhất, loại X...) → chỉ nói về mục đó.
   - Hỏi liệt kê/tổng hợp → mới liệt kê đầy đủ.
   - Hỏi số lượng của một nhóm/danh sách (ví dụ: "có mấy giá trị cốt lõi?") → trả lời con số trước, rồi liệt kê tên các mục nếu tài liệu có thông tin.

3. CHI TIẾT: Khai thác đầy đủ thông tin từ tài liệu (định nghĩa, giải thích, ví dụ, số liệu).

4. KHÔNG LẶP LẠI:
   Mỗi luận điểm phải có thông tin RIÊNG BIỆT từ tài liệu.
   KHÔNG sao chép cùng một câu/đoạn văn từ một nguồn cho nhiều ý khác nhau.
   Nếu nguồn chỉ có thông tin tổng quát cho một mục cụ thể, hãy trình bày đúng những gì có — không bịa thêm.

5. ĐỊNH DẠNG: Dùng ### tiêu đề, **in đậm** từ khóa, danh sách -.
   Đi thẳng vào nội dung, không thêm tiêu đề dẫn nhập thừa.
   Dùng 'mình' và 'bạn'.
   Nếu toàn bộ nội dung có thể kết luận thành ý chính, hãy tóm tắt ý chính đó ở cuối phần trả lời (sau khi đã trình bày chi tiết), chỉ cần có nhiều nhất 1 kết luận ngắn gọn, rõ ràng, KHÔNG thêm bất kỳ lời dẫn nhập nào cho phần kết luận này.\
"""

_QA_NO_CONTEXT_SYSTEM_PROMPT = """\
Bạn là Trợ lý tri thức VTAca - trợ lý hội thoại tiếng Việt của Viettel Academy.

QUY TẮC TRẢ LỜI KHI KHÔNG CÓ TÀI LIỆU THAM KHẢO:
- Trả lời tự nhiên, ngắn gọn, thân thiện theo ngữ cảnh hội thoại.
- Với greeting/chit-chat (ví dụ: "xin chào", "cảm ơn"), trả lời trực tiếp như trợ lý thông thường.
- KHÔNG nhắc tới "Chunk", "tài liệu tham khảo", hoặc lý do thiếu tài liệu.
- KHÔNG bịa dữ kiện chuyên môn; nếu người dùng hỏi kiến thức cần kiểm chứng, nói rõ cần cung cấp tài liệu/chủ đề cụ thể.
- Dùng xưng hô "mình" và "bạn".
"""

_SYSTEM_PROMPTS: dict[str, str] = {
    "qa":          _QA_SYSTEM_PROMPT,
}

_USER_PROMPT_TASK_LABELS: dict[str, str] = {
    "qa":          "Phân tích và trả lời theo định dạng sau",
}


# ─── Message builder ──────────────────────────────────────────────────────────

_HISTORY_LIMIT = {"qa": 8, "outline": 4, "script": 4, "quiz": 4, "summary_doc": 4}
_ASSISTANT_PREVIEW = {"qa": 1200, "outline": 500, "script": 500, "quiz": 500, "summary_doc": 500}


def _build_messages(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
    output_mode: str = "qa",
) -> list:
    # Drop the last message if it's the current user question already captured in YÊU CẦU.
    history = list(history_messages)
    if history and history[-1].role == "user" and history[-1].content.strip() == question.strip():
        history = history[:-1]

    limit = _HISTORY_LIMIT.get(output_mode, 8)
    preview_len = _ASSISTANT_PREVIEW.get(output_mode, 1200)
    history = history[-limit:]

    def _fmt(msg: ChatMessage) -> str:
        content = msg.content
        if msg.role == "assistant" and len(content) > preview_len:
            content = content[:preview_len].rstrip() + "\n... [đã rút gọn]"
        return f"{msg.role.upper()}: {content}"

    history_block = "\n".join(_fmt(msg) for msg in history)
    has_context = len(context_docs) > 0
    context_block = _build_context_block(context_docs) if has_context else ""
    system_content = _SYSTEM_PROMPTS.get(output_mode, _QA_SYSTEM_PROMPT)
    if output_mode == "qa" and not has_context:
        system_content = _QA_NO_CONTEXT_SYSTEM_PROMPT
    task_label = _USER_PROMPT_TASK_LABELS.get(output_mode, _USER_PROMPT_TASK_LABELS["qa"])

    if has_context:
        user_content = (
            "=== TÀI LIỆU THAM KHẢO ===\n"
            f"{context_block}\n\n"
            "=== LỊCH SỬ TRÒ CHUYỆN ===\n"
            f"{history_block or 'Chưa có.'}\n\n"
            "=== YÊU CẦU ===\n"
            f"{question}\n\n"
            f"{task_label}:"
        )
    else:
        user_content = (
            "=== LỊCH SỬ TRÒ CHUYỆN ===\n"
            f"{history_block or 'Chưa có.'}\n\n"
            "=== YÊU CẦU ===\n"
            f"{question}\n\n"
            "Trả lời hội thoại trực tiếp:"
        )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]


# ─── Generation entry points ──────────────────────────────────────────────────

def generate_answer(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
    output_mode: str = "qa",
) -> str:
    with _timed_query_step("load_chat_llm", event_prefix="generate_answer_load_llm"):
        llm = get_llm()

    with _timed_query_step(
        "build_prompt",
        event_prefix="generate_answer_build_prompt",
        details={"history_count": len(history_messages), "context_doc_count": len(context_docs)},
    ):
        messages = _build_messages(question, context_docs, history_messages, output_mode)

    with _timed_query_step(
        "invoke_chat_llm",
        event_prefix="generate_answer_invoke_llm",
        details={"question_preview": _preview_text(question), "context_doc_count": len(context_docs)},
    ):
        response = llm.invoke(messages)

    return str(response.content) if hasattr(response, "content") else str(response)


def generate_answer_stream(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
    output_mode: str = "qa",
) -> Iterator[str]:
    with _timed_query_step("load_chat_llm", event_prefix="generate_answer_load_llm"):
        llm = get_llm()

    with _timed_query_step(
        "build_prompt",
        event_prefix="generate_answer_build_prompt",
        details={"history_count": len(history_messages), "context_doc_count": len(context_docs)},
    ):
        messages = _build_messages(question, context_docs, history_messages, output_mode)

    _emit_query_progress(
        "[chat.query] Start stream response: output_mode=%s, context_doc_count=%d",
        output_mode,
        len(context_docs),
    )

    for chunk in llm.stream(messages):
        content = str(chunk.content) if hasattr(chunk, "content") else str(chunk)
        if content:
            yield content


def parse_sources(
    raw_json: str | None,
) -> list[dict[str, int | str | float | dict | None]]:
    if not raw_json:
        return []
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return []
