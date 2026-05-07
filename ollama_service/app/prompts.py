from __future__ import annotations


INDEXING_INSTRUCTION = """
Bạn là hệ thống xử lý tài liệu cho RAG indexing.

Hãy phân tích văn bản và trả về JSON hợp lệ, không giải thích thêm.

Schema bắt buộc:
{
  "summary": "Tóm tắt ngắn 3-5 câu",
  "hyq": [
    "Câu hỏi giả định 1",
    "Câu hỏi giả định 2",
    "Câu hỏi giả định 3"
  ],
  "metadata": {
    "title": null,
    "topic": null,
    "keywords": [],
    "document_type": null,
    "department_or_unit": null,
    "date": null,
    "people": [],
    "risk_level": "low"
  },
  "language": "vi"
}
"""


def build_indexing_prompt(instruction: str, text: str) -> str:
    return f"""{instruction.strip()}

Văn bản cần xử lý:
\"\"\"
{text}
\"\"\"
"""
