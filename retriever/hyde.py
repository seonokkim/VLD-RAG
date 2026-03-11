"""Lightweight HyDE helpers for hybrid retrieval workflows."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional


logger = logging.getLogger(__name__)


class HyDEGenerator:
    """Generate hypothetical documents from a query for dense retrieval."""

    def __init__(
        self,
        generator: Optional[Callable[..., Any]] = None,
        n: int = 1,
        instruction: Optional[str] = None,
        include_original_query: bool = False,
    ):
        self.generator = generator
        self.n = max(1, n)
        self.instruction = instruction
        self.include_original_query = include_original_query

    def generate(
        self,
        query: str,
        n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[str]:
        """Generate one or more hypothetical documents for the query."""
        target_n = max(1, n or self.n)
        active_instruction = instruction or self.instruction

        if self.generator is None:
            outputs = self._generate_template_documents(query, target_n, active_instruction)
        else:
            outputs = self._normalize_generated_output(
                self._call_generator(query=query, n=target_n, instruction=active_instruction)
            )

        if self.include_original_query:
            outputs.append(query)

        deduplicated: List[str] = []
        seen = set()
        for item in outputs:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduplicated.append(normalized)

        return deduplicated or [query]

    def _call_generator(self, query: str, n: int, instruction: Optional[str]) -> Any:
        """Call an injected HyDE generator object or function."""
        candidate = self.generator

        if hasattr(candidate, "generate"):
            return candidate.generate(query=query, n=n, instruction=instruction)
        if hasattr(candidate, "expand"):
            return candidate.expand(query=query, n=n, instruction=instruction)

        try:
            return candidate(query=query, n=n, instruction=instruction)
        except TypeError:
            try:
                return candidate(query, n=n, instruction=instruction)
            except TypeError:
                return candidate(query)

    def _normalize_generated_output(self, value: Any) -> List[str]:
        """Normalize supported generator return types into a string list."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        if isinstance(value, tuple):
            return [item for item in value if isinstance(item, str)]
        return [str(value)]

    def _generate_template_documents(
        self,
        query: str,
        n: int,
        instruction: Optional[str],
    ) -> List[str]:
        """Fallback deterministic HyDE generation without external LLMs."""
        prompt_hint = (
            instruction.strip()
            if instruction
            else "Write a concise passage that would likely contain the answer."
        )
        variants = [
            (
                f"{prompt_hint}\n\n"
                f"Question: {query}\n\n"
                "Hypothetical passage:\n"
                f"This document describes the context needed to answer the question '{query}'. "
                "It highlights the key entities, values, page-level evidence, and supporting details "
                "that a relevant source chunk would likely contain."
            ),
            (
                f"{prompt_hint}\n\n"
                f"Question: {query}\n\n"
                "Hypothetical passage:\n"
                "A relevant section would restate the question in declarative form, mention the main "
                "document entities, and include the most likely evidence-bearing terms, headings, "
                "tables, or figure references connected to the answer."
            ),
            (
                f"{prompt_hint}\n\n"
                f"Question: {query}\n\n"
                "Hypothetical passage:\n"
                "The answer likely appears in a chunk that contains semantically similar wording, "
                "supporting descriptions, and adjacent context that explains the requested fact "
                "with enough detail to verify it."
            ),
        ]
        if n <= len(variants):
            return variants[:n]
        return [variants[i % len(variants)] for i in range(n)]
