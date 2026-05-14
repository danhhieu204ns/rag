"""
LLM-based query orchestrator — produces OrchestrationPlan via LLM instead of regex.

Orchestrator mode switching:
  - "rule": Use rule-based regex + heuristics (original behavior, < 1ms)
  - "llm": Use LLM planner with fallback to rule-based on error/timeout
  - "hybrid": Use LLM for most cases, but keep rule-based for high-precision cases
             like document code lookup

LLM responses are strictly validated and clamped to safe ranges to ensure
robustness even with model hallucination or miscalibration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from ...core.settings import settings

logger = logging.getLogger(__name__)


# ─── LLM Prompt ───────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an expert query orchestrator for a RAG system. Your task is to analyze \
a user query and produce a JSON orchestration plan that guides hybrid retrieval \
and output generation.

# JSON Response Format

Output ONLY valid JSON, no markdown, no explanations. Structure:

{
  "query_type": "string describing the query intent",
  "output_mode": "qa" | "outline" | "script" | "quiz" | "summary_doc",
  "strategy": "keyword_heavy" | "vector_heavy" | "balanced" | "broad",
  "top_k_multiplier": number between 1.0 and 5.0,
  "vector_rrf_weight": number between 0.1 and 3.0,
  "keyword_rrf_weight": number between 0.1 and 3.0,
  "use_reranker": boolean,
  "candidate_pool_delta": integer between 0 and 20,
  "max_iterations": integer between 1 and 3,
  "expand_query": boolean,
  "requires_context": boolean,
  "coverage_mode": "topk" | "section_coverage" | "document_coverage",
  "confidence": number between 0.0 and 1.0,
  "signals": ["signal1", "signal2"],
  "reason": "brief explanation of the orchestration plan"
}

# Classification Rules

## output_mode Selection
- "qa": General question-answering (default for most queries)
- "outline": If user asks to "create outline", "make outline", "structure as outline"
- "script": If user asks to "write script", "create lesson plan", "make teaching script"
- "quiz": If user asks to "create quiz", "make test", "generate questions"
- "summary_doc": If user asks to "summarize entire document", "summarize all", "document summary"

## strategy Selection
- "keyword_heavy": Document codes, legal references, specific procedures, named entities
- "vector_heavy": Conceptual queries, definitions, abstract concepts, semantic understanding
- "balanced": Mixed keyword+semantic, comparative, procedural questions
- "broad": Comparative queries ("compare X vs Y"), listing queries ("list all types of...")

## Special Cases

### Document Code / Legal Reference Detection
- Query contains patterns like "Decision 123/2021", "Directive 456-VN", regulation codes
- Action: strategy="keyword_heavy", use_reranker=false, expand_query=false

### Contextual Follow-up
- Very short query (≤2 words) like "nó là gì?" ("what is it?"), "vậy còn cái này?"
- Action: requires_context=true, strategy="vector_heavy", vector_rrf_weight should be high (2.0+)

### Definition / Conceptual
- Query contains "definition of", "what is", "bao gồm những gì"
- Action: strategy="vector_heavy", vector_rrf_weight=1.8, keyword_rrf_weight=0.5

### Comparative
- Query contains "compare", "so sánh", "difference between", "giống vs khác"
- Action: strategy="broad", expand_query=true, top_k_multiplier should be higher (2.0+)

### Procedural / How-to
- Query contains "how to", "lam the nao", "quy trinh", "buoc", "thu tuc"
- Action: strategy="vector_heavy" or "balanced", max_iterations=1 or 2

### Factual / Quantitative
- Query contains numbers, dates, statistics: "bao nhieu", "nam nao", "ty le", "muc phat"
- Action: strategy="balanced", keyword_rrf_weight should be higher (1.2+)

### Listing / Comprehensive
- Query asks for complete list, all items, enumeration: "liet ke", "tat ca", "nhung loai nao"
- Action: top_k_multiplier should be higher (2.5+), strategy="vector_heavy" or "balanced", 
  coverage_mode="section_coverage" or "document_coverage"

### Summary / Overview
- Query asks for summary of document/section: "tom tat", "tong quan", "noi dung chinh"
- Action: coverage_mode="document_coverage", top_k_multiplier=3.0+, strategy="vector_heavy"

## confidence Score
- 0.95-1.0: Very clear query intent (document codes, definition questions)
- 0.75-0.94: Clear intent with minor ambiguity (comparative, procedural)
- 0.60-0.74: Moderate ambiguity (generic queries, multiple interpretations possible)
- 0.40-0.59: High ambiguity (very short, vague, conflicting signals)
- < 0.40: Not confident, may want rule-based fallback

## Clamping & Defaults
- top_k_multiplier: Always between 1.0 and 5.0, default 1.0
- vector_rrf_weight: Always between 0.1 and 3.0, default 1.0
- keyword_rrf_weight: Always between 0.1 and 3.0, default 1.0
- candidate_pool_delta: Always between 0 and 20, default 0
- max_iterations: Always between 1 and 3, default 1
- confidence: Always between 0.0 and 1.0, default 0.5

## signals Array
- Include any relevant detection signals: "document_code", "definition", "comparative", "procedural", etc.
- May include "low_confidence" if < 0.5
- Never leave empty; always include at least one signal describing the query type

# Examples

Query: "Quyết định 123/2021 là gì?"
{
  "query_type": "document_code_lookup",
  "output_mode": "qa",
  "strategy": "keyword_heavy",
  "top_k_multiplier": 1.0,
  "vector_rrf_weight": 0.2,
  "keyword_rrf_weight": 2.5,
  "use_reranker": false,
  "candidate_pool_delta": 0,
  "max_iterations": 1,
  "expand_query": false,
  "requires_context": false,
  "coverage_mode": "topk",
  "confidence": 0.98,
  "signals": ["document_code", "definition"],
  "reason": "Clear document code reference detected. Use keyword-heavy retrieval."
}

Query: "So sánh CNN và ViT"
{
  "query_type": "comparative",
  "output_mode": "qa",
  "strategy": "broad",
  "top_k_multiplier": 2.0,
  "vector_rrf_weight": 1.2,
  "keyword_rrf_weight": 1.0,
  "use_reranker": true,
  "candidate_pool_delta": 10,
  "max_iterations": 2,
  "expand_query": true,
  "requires_context": false,
  "coverage_mode": "topk",
  "confidence": 0.85,
  "signals": ["comparative"],
  "reason": "Comparative query detected. Use broad strategy with query expansion."
}

Query: "Tóm tắt toàn bộ tài liệu"
{
  "query_type": "structured_summary_doc",
  "output_mode": "summary_doc",
  "strategy": "vector_heavy",
  "top_k_multiplier": 4.0,
  "vector_rrf_weight": 1.8,
  "keyword_rrf_weight": 0.5,
  "use_reranker": false,
  "candidate_pool_delta": 10,
  "max_iterations": 1,
  "expand_query": false,
  "requires_context": false,
  "coverage_mode": "document_coverage",
  "confidence": 0.92,
  "signals": ["summary", "output_mode:summary_doc"],
  "reason": "Document summary requested. Use wide coverage mode."
}

Query: "Nó là gì?"
{
  "query_type": "contextual_followup",
  "output_mode": "qa",
  "strategy": "vector_heavy",
  "top_k_multiplier": 1.0,
  "vector_rrf_weight": 2.2,
  "keyword_rrf_weight": 0.2,
  "use_reranker": false,
  "candidate_pool_delta": 0,
  "max_iterations": 1,
  "expand_query": false,
  "requires_context": true,
  "coverage_mode": "topk",
  "confidence": 0.7,
  "signals": ["short_or_followup", "requires_context"],
  "reason": "Short follow-up question. Requires conversation context."
}
"""


class LLMOrchestratorClient:
    """Client for LLM-based query orchestration planning."""

    def __init__(self):
        self.model = settings.orchestrator_llm_model
        self.temperature = settings.orchestrator_llm_temperature
        self.max_tokens = settings.orchestrator_llm_max_tokens
        self.timeout_seconds = settings.orchestrator_llm_timeout_seconds
        self.ollama_base_url = settings.ollama_base_url.rstrip("/")
        self.ollama_api_key = settings.ollama_api_key
        
    async def plan(
        self,
        query: str,
        base_top_k: int,
        output_mode_override: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate an orchestration plan via LLM.
        
        Returns:
            dict with orchestration plan, or None on failure (will fallback to rule-based)
        """
        if not self.ollama_base_url or not self.ollama_api_key:
            logger.warning("[llm_orchestrator] OLLAMA_BASE_URL or OLLAMA_API_KEY not configured")
            return None
            
        started_at = time.perf_counter()
        
        # Build user prompt with query and base_top_k
        user_prompt = self._build_user_prompt(query, base_top_k, output_mode_override)
        
        try:
            plan_dict = await self._call_llm(user_prompt)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            
            logger.info(
                "[llm_orchestrator] plan generated elapsed_ms=%.2f confidence=%s",
                elapsed_ms,
                plan_dict.get("confidence", "?"),
            )
            return plan_dict
            
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.warning("[llm_orchestrator] timeout after %.2fms", elapsed_ms)
            return None
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.warning(
                "[llm_orchestrator] error after %.2fms: %s %s",
                elapsed_ms,
                type(exc).__name__,
                exc,
            )
            return None

    async def _call_llm(self, user_prompt: str) -> dict[str, Any]:
        """Call LLM with structured prompt, parse and return JSON."""
        timeout = httpx.Timeout(self.timeout_seconds, connect=5.0)
        headers = {"x-api-key": self.ollama_api_key}
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "stream": False,
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                headers=headers,
            )
        
        if response.status_code >= 400:
            raise RuntimeError(
                f"LLM returned HTTP {response.status_code}: {response.text[:200]}"
            )
        
        result = response.json()
        raw_response = result.get("message", {}).get("content", "")
        
        # Try to parse JSON from response
        json_str = self._extract_json(raw_response)
        plan_dict = json.loads(json_str)
        
        return plan_dict

    def plan_sync(
        self,
        query: str,
        base_top_k: int,
        output_mode_override: str | None = None,
    ) -> dict[str, Any] | None:
        """Synchronous wrapper for plan() to be used from sync code paths."""
        if not self.ollama_base_url or not self.ollama_api_key:
            logger.warning("[llm_orchestrator] OLLAMA_BASE_URL or OLLAMA_API_KEY not configured")
            return None

        user_prompt = self._build_user_prompt(query, base_top_k, output_mode_override)
        timeout = httpx.Timeout(self.timeout_seconds, connect=5.0)
        headers = {"x-api-key": self.ollama_api_key}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "stream": False,
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{self.ollama_base_url}/api/chat", json=payload, headers=headers)

        if response.status_code >= 400:
            logger.warning("[llm_orchestrator] LLM HTTP %s: %s", response.status_code, response.text[:200])
            return None

        result = response.json()
        raw_response = result.get("message", {}).get("content", "")
        try:
            json_str = self._extract_json(raw_response)
            plan_dict = json.loads(json_str)
            return plan_dict
        except Exception:
            logger.exception("[llm_orchestrator] failed to parse LLM JSON")
            return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON object from LLM response (may contain explanation)."""
        # Try to find JSON object in response
        text = text.strip()
        
        # If starts with { and ends with }, try direct parse
        if text.startswith("{") and text.endswith("}"):
            return text
        
        # Try to find JSON object pattern
        start_idx = text.find("{")
        if start_idx == -1:
            raise ValueError("No JSON object found in LLM response")
        
        # Find matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
        
        raise ValueError("Unmatched braces in LLM JSON response")

    @staticmethod
    def _build_user_prompt(
        query: str,
        base_top_k: int,
        output_mode_override: str | None = None,
    ) -> str:
        """Build user prompt for LLM orchestrator."""
        base_prompt = f"""\
Analyze this query and produce an orchestration plan:

Query: {query}

Base top_k for retrieval: {base_top_k}
"""
        
        if output_mode_override:
            base_prompt += f"Output mode override: {output_mode_override}\n"
        
        base_prompt += "\nRespond with ONLY the JSON orchestration plan, no other text."
        return base_prompt


# Global client instance
_client: LLMOrchestratorClient | None = None


def get_llm_orchestrator_client() -> LLMOrchestratorClient:
    """Get or create global LLM orchestrator client."""
    global _client
    if _client is None:
        _client = LLMOrchestratorClient()
    return _client
