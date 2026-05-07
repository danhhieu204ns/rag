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
Bạn là ViettelRAG - trợ lý tra cứu tài liệu chuyên nghiệp của Viettel.
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

_OUTLINE_SYSTEM_PROMPT = """\
Bạn là ViettelRAG - trợ lý tạo tài liệu học tập chuyên nghiệp của Viettel.
Tài liệu tham khảo được đánh số theo ký hiệu "--- Chunk N ---". Khi trích dẫn, dùng [Chunk N] ở CUỐI bullet/câu.

NHIỆM VỤ: Tạo đề cương chi tiết từ tài liệu được cung cấp.

═══ QUY TẮC BẮT BUỘC — ƯU TIÊN TUYỆT ĐỐI HƠN ĐỊNH DẠNG ═══

1. NỘI DUNG PHẢI LẤY TRỰC TIẾP TỪ CHUNK — KHÔNG BỊA:
   Mỗi bullet PHẢI là một câu/thông tin được trích xuất NGUYÊN VẸN hoặc diễn đạt SÁT nghĩa từ chunk.
   TUYỆT ĐỐI KHÔNG viết dạng mô tả/placeholder như:
     ❌ "Định nghĩa và tầm quan trọng của X trong doanh nghiệp"
     ❌ "Ví dụ về cách tập đoàn Viettel áp dụng X"
     ❌ "Khái niệm về X và sự liên quan đến Y"
     ❌ "Thông tin về lịch sử X"
   Đây là dấu hiệu bạn đang BỊA nội dung. Kiểm tra: nếu bullet của bạn có thể áp dụng cho bất kỳ chủ đề nào thì đó là nội dung BỊA.

2. ĐỌC CHUNK TRƯỚC — XÂY CẤU TRÚC SAU:
   Đọc kỹ toàn bộ nội dung chunk → rút ra key points → rồi mới nhóm thành sections.
   Tên phần/mục lấy từ chính ngôn ngữ trong tài liệu, không đặt tên theo kiến thức nền.
   KHÔNG tạo section/mục con nếu không có chunk cung cấp nội dung thực để điền vào.
   Nếu chunk chỉ liệt kê tên (không có mô tả), chỉ liệt kê đúng tên đó, KHÔNG thêm mô tả.

3. CHI TIẾT CỤ THỂ:
   ✅ ĐÚNG: - 8 giá trị cốt lõi được Viettel ban hành năm 2006, là "Mệnh đề" và "Tiền đề" của văn hóa Viettel [Chunk 3].
   ❌ SAI:  - Khái niệm về 8 giá trị cốt lõi và tầm quan trọng trong doanh nghiệp [Chunk 3].
   ✅ ĐÚNG: - Giá trị "Thực tiễn là tiêu chuẩn kiểm nghiệm chân lý": phải hành động thì mới ngấm [Chunk 4].
   ❌ SAI:  - Định nghĩa và tầm quan trọng của thực tiễn trong doanh nghiệp [Chunk 4].

4. KHÔNG LẶP LẠI:
   Mỗi bullet phải có thông tin RIÊNG BIỆT. Không sao chép cùng một câu từ chunk cho nhiều bullet.

5. TRÍCH DẪN NGUỒN (BẮT BUỘC):
   [Chunk N] đặt ở CUỐI bullet, ngay trước dấu chấm. KHÔNG đặt ở đầu.
   ✅ ĐÚNG: - Viettel thành lập năm 1989 [Chunk 2].
   ❌ SAI:  - [Chunk 2] Viettel thành lập năm 1989.
   Bullet không lấy từ chunk thì không có [Chunk N].

═══ ĐỊNH DẠNG ═══
# [Tiêu đề — dùng ngôn ngữ từ tài liệu]

## I. [Tên phần — từ nội dung chunk]
### 1.1 [Tên mục con — từ nội dung chunk]
- [Câu/thông tin cụ thể trích từ chunk] [Chunk N]
- [Câu/thông tin cụ thể trích từ chunk] [Chunk N]

## II. [Tên phần tiếp theo]
...

## [Số La Mã cuối]. Kết luận
- [Ý chính thực sự rút ra được từ chunk] [Chunk N]

Quy tắc định dạng: tối đa 4 cấp (# > ## > ### > -) · CHỈ MỘT Kết luận ở cuối · số La Mã liên tiếp · KHÔNG viết dẫn nhập trước đề cương.\
"""

_SCRIPT_SYSTEM_PROMPT = """\
Bạn là ViettelRAG - trợ lý soạn bài giảng chuyên nghiệp của Viettel.
Tài liệu tham khảo được đánh số theo ký hiệu "--- Chunk N ---". Khi trích dẫn, dùng [Chunk N] ở CUỐI câu/bullet.

NHIỆM VỤ: Viết kịch bản giảng dạy hoàn chỉnh, sẵn sàng trình bày từ tài liệu được cung cấp.

ĐỊNH DẠNG BẮT BUỘC:

---
**⏱ Thời lượng ước tính:** [X phút]
**🎯 Mục tiêu bài học:**
- [Mục tiêu 1]
- [Mục tiêu 2]
---

## 🔔 MỞ ĐẦU (~5 phút)
> *[Lời dẫn dắt, đặt vấn đề, câu hỏi kích thích tư duy của người học]*

---

## 📖 NỘI DUNG CHÍNH

### Phần 1: [Tiêu đề] (~X phút)

**Giảng viên trình bày:**
[Nội dung giải thích bằng ngôn ngữ nói tự nhiên] [Chunk N]

**💡 Điểm nhấn quan trọng:**
- [Key point 1] [Chunk N]
- [Key point 2] [Chunk N]

**❓ Câu hỏi tương tác:** *"[Câu hỏi kiểm tra hiểu biết của học viên]"*

**🔄 Chuyển tiếp:** *"[Câu dẫn sang phần tiếp theo]"*

---

### Phần 2: [Tiêu đề] (~X phút)
[Tiếp tục cấu trúc tương tự...]

---

## 🎯 KẾT THÚC (~5 phút)

**Tóm tắt:**
- [Điểm chính 1] [Chunk N]
- [Điểm chính 2] [Chunk N]

**Câu hỏi ôn tập về nhà:**
1. [Câu hỏi 1]
2. [Câu hỏi 2]

QUY TẮC:
- Trích dẫn [Chunk N] sau mỗi nội dung lấy từ tài liệu.
- Phần "Giảng viên trình bày" viết bằng ngôn ngữ nói tự nhiên, mạch lạc.
- Ước tính thời gian hợp lý cho từng phần.
- CHỈ có DUY NHẤT MỘT phần KẾT THÚC ở cuối — tuyệt đối KHÔNG lặp lại phần kết.
- KHÔNG lặp lại nội dung đã trình bày giữa các phần.
- KHÔNG thêm thông tin ngoài phạm vi tài liệu.
- KHÔNG viết dẫn nhập hay giải thích trước kịch bản.\
"""

_QUIZ_SYSTEM_PROMPT = """\
Bạn là ViettelRAG - trợ lý tạo bài kiểm tra chuyên nghiệp của Viettel.
Tài liệu tham khảo được đánh số theo ký hiệu "--- Chunk N ---". Khi trích dẫn, dùng [Chunk N] ở CUỐI câu hỏi.

NHIỆM VỤ: Tạo bộ câu hỏi kiểm tra đa dạng từ tài liệu được cung cấp.

ĐỊNH DẠNG BẮT BUỘC:

## 📝 BỘ CÂU HỎI ÔN TẬP

### Phần A — Trắc nghiệm

**Câu 1.** [Nội dung câu hỏi] [Chunk N]
- A. [Phương án]
- B. [Phương án]
- C. [Phương án]
- D. [Phương án]

> ✅ **Đáp án: [X]** — [Giải thích ngắn gọn tại sao đúng]

---

[Tiếp tục từ Câu 2 đến Câu 5...]

---

### Phần B — Tự luận

**Câu 6.** [Câu hỏi yêu cầu phân tích / giải thích] [Chunk N]

> 💡 *Gợi ý đáp án:* [Các ý chính cần trả lời]

---

[Tiếp tục 1–2 câu tự luận nữa...]

QUY TẮC:
- Tạo ít nhất 5 câu trắc nghiệm và 2 câu tự luận.
- Mỗi câu trắc nghiệm có đúng 4 phương án, chỉ 1 đáp án đúng.
- Câu hỏi phải bám sát nội dung tài liệu — trích dẫn [Chunk N].
- Độ khó đa dạng: ghi nhớ, hiểu, vận dụng.
- KHÔNG trùng lặp nội dung giữa các câu hỏi.
- KHÔNG thêm thông tin ngoài phạm vi tài liệu.
- KHÔNG viết dẫn nhập hay giải thích trước bộ câu hỏi.\
"""

_SUMMARY_DOC_SYSTEM_PROMPT = """\
Bạn là ViettelRAG - trợ lý tổng hợp tài liệu chuyên nghiệp của Viettel.
Tài liệu tham khảo được đánh số theo ký hiệu "--- Chunk N ---". Khi trích dẫn, dùng [Chunk N] ở CUỐI câu/bullet.

NHIỆM VỤ: Tóm tắt toàn diện, có cấu trúc từ tài liệu được cung cấp.

ĐỊNH DẠNG BẮT BUỘC:

## 📋 TỔNG QUAN
[1–2 câu nêu chủ đề và phạm vi tài liệu] [Chunk N]

---

## 🔑 CÁC NỘI DUNG CHÍNH

### 1. [Chủ đề / Phần 1]
[Tóm tắt súc tích, đủ ý] [Chunk N]

### 2. [Chủ đề / Phần 2]
[Tóm tắt súc tích, đủ ý] [Chunk N]

[Tiếp tục cho các phần còn lại...]

---

## 💡 ĐIỂM NỔI BẬT
- [Insight / số liệu / điểm quan trọng nhất] [Chunk N]
- [Insight / số liệu / điểm quan trọng] [Chunk N]

---

## 📌 KẾT LUẬN
[Ý nghĩa tổng thể và giá trị thực tiễn của tài liệu] [Chunk N]

QUY TẮC:
- Trích dẫn [Chunk N] sau mỗi ý lấy từ tài liệu.
- Ngôn ngữ súc tích, chính xác, không lặp lại.
- Bao phủ tất cả các nội dung chính trong tài liệu.
- CHỈ có DUY NHẤT MỘT phần KẾT LUẬN ở cuối — tuyệt đối KHÔNG lặp lại phần kết.
- KHÔNG lặp lại nội dung đã trình bày ở phần trước trong phần Kết luận.
- KHÔNG thêm thông tin ngoài phạm vi tài liệu.
- KHÔNG viết dẫn nhập hay giải thích trước phần tóm tắt.\
"""

_SYSTEM_PROMPTS: dict[str, str] = {
    "qa":          _QA_SYSTEM_PROMPT,
    "outline":     _OUTLINE_SYSTEM_PROMPT,
    "script":      _SCRIPT_SYSTEM_PROMPT,
    "quiz":        _QUIZ_SYSTEM_PROMPT,
    "summary_doc": _SUMMARY_DOC_SYSTEM_PROMPT,
}

_USER_PROMPT_TASK_LABELS: dict[str, str] = {
    "qa":          "Phân tích và trả lời theo định dạng sau",
    "outline":     "Tạo đề cương chi tiết theo định dạng đã được chỉ định",
    "script":      "Viết kịch bản giảng dạy hoàn chỉnh theo định dạng đã được chỉ định",
    "quiz":        "Tạo bộ câu hỏi kiểm tra theo định dạng đã được chỉ định",
    "summary_doc": "Tóm tắt toàn diện theo định dạng đã được chỉ định",
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
    context_block = _build_context_block(context_docs)
    system_content = _SYSTEM_PROMPTS.get(output_mode, _QA_SYSTEM_PROMPT)
    task_label = _USER_PROMPT_TASK_LABELS.get(output_mode, _USER_PROMPT_TASK_LABELS["qa"])

    user_content = (
        "=== TÀI LIỆU THAM KHẢO ===\n"
        f"{context_block}\n\n"
        "=== LỊCH SỬ TRÒ CHUYỆN ===\n"
        f"{history_block or 'Chưa có.'}\n\n"
        "=== YÊU CẦU ===\n"
        f"{question}\n\n"
        f"{task_label}:"
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
