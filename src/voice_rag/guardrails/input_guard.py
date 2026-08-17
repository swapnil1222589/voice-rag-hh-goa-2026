"""Input guardrail — validates and filters incoming queries.

Checks:
1. Length constraints (too short = noise, too long = abuse)
2. Unsafe content detection (profanity, violence, self-harm, exploitation)
3. PII detection (email, phone, SSN-like patterns)
4. Injection attempts (prompt-injection patterns like "ignore previous instructions")
5. Off-topic detection (handled at retrieval time via relevance threshold)
"""
from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Enums ───────────────────────────────────────────────────────
class GuardrailAction(str, enum.Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"       # allow but flag


# ── Unsafe content patterns ────────────────────────────────────
# We use a keyword-based approach — fast, no external API call, no false-positive model.
# This is deliberately conservative: we block clearly harmful content and let the
# output guardrail handle subtler cases.

_UNSAFE_PATTERNS = [
    # Violence / harm
    r"\b(kill|murder|assassinate|massacre|shoot|stab|bomb|behead)\b.*\b(how to|ways to|instructions?)\b",
    r"\bhow to (make|build|create).*(bomb|explosive|weapon|poison|nerve gas|chemical weapon)\b",
    r"\b(mass shooting|school shooting|terrorist attack).*(how to|plan|execute)\b",
    # Self-harm
    r"\b(how to (kill|hurt|harm) (myself|yourself)|suicide methods|self.harm techniques)\b",
    r"\b(best way to|painless way to|quickest way to).*(die|kill myself|end it all)\b",
    # Exploitation
    r"\b(child|minor|underage|teen).*(nude|sexual|exploit|groom)\b",
    r"\b(csam|child abuse|pedophil)\b",
]

# Prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) (instructions?|prompts?|rules?)",
    r"disregard (all )?(previous|prior) (instructions?|context)",
    r"you are (now )?(a|an) (different|new|jailbroken|unrestricted)",
    r"(reveal|show|print|output) (your |the )?(system|initial|original) prompt",
    r"\b(DAN|developer mode|jailbreak)\b",
]

# PII patterns (lightweight)
_PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
    (r'\b\d{10}\b', "phone"),
    (r'\b\d{3}-\d{2}-\d{4}\b', "ssn"),
    (r'\b\d{16}\b', "credit_card"),
]

_COMPILED_UNSAFE = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_COMPILED_PII = [(re.compile(p), name) for p, name in _PII_PATTERNS]


@dataclass
class GuardrailResult:
    action: GuardrailAction
    reason: str = ""
    cleaned_query: str = ""
    flags: list = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []

    @property
    def allowed(self) -> bool:
        return self.action == GuardrailAction.ALLOW

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "flags": self.flags,
        }


class InputGuardrail:
    """Validates incoming queries before they reach the retrieval pipeline."""

    def __init__(
        self,
        max_query_length: int = 500,
        min_query_length: int = 2,
    ):
        self.max_len = max_query_length
        self.min_len = min_query_length

    def check(self, query: str) -> GuardrailResult:
        """Run all input checks.  Returns ALLOW / BLOCK / WARN."""
        flags: list = []
        q = (query or "").strip()

        # 1. Empty / length
        if not q:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="Empty query",
                flags=["empty"],
            )
        if len(q) < self.min_len:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Query too short ({len(q)} chars, min {self.min_len})",
                flags=["too_short"],
            )
        if len(q) > self.max_len:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason=f"Query too long ({len(q)} chars, max {self.max_len})",
                flags=["too_long"],
            )

        # 2. Unsafe content
        for pat in _COMPILED_UNSAFE:
            if pat.search(q):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Query contains unsafe or harmful content",
                    flags=["unsafe_content"],
                )

        # 3. Prompt injection
        for pat in _COMPILED_INJECTION:
            if pat.search(q):
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Prompt injection attempt detected",
                    flags=["injection_attempt"],
                )

        # 4. PII — warn but don't block (user might legitimately ask about a concept)
        for pat, name in _COMPILED_PII:
            if pat.search(q):
                flags.append(f"pii_{name}")

        # 5. Gibberish detection — if the query has no recognizable words
        # (e.g., STT produced noise), flag it
        words = q.split()
        if len(words) >= 3:
            alpha_ratio = sum(1 for w in words if any(c.isalpha() for c in w)) / len(words)
            if alpha_ratio < 0.3:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    reason="Query appears to be gibberish (STT noise?)",
                    flags=["gibberish"],
                )

        action = GuardrailAction.WARN if flags else GuardrailAction.ALLOW
        return GuardrailResult(
            action=action,
            reason="PII detected — proceeding with caution" if flags else "OK",
            cleaned_query=q,
            flags=flags,
        )

    def check_relevance(
        self,
        retrieval_results: list,
        threshold: float = 0.25,
    ) -> GuardrailResult:
        """Check if the retrieved context is relevant enough to answer.

        Called after retrieval but before generation — if the best retrieval
        score is below the threshold, the system should refuse rather than
        hallucinate.
        """
        if not retrieval_results:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                reason="No relevant context found — refusing to answer to avoid hallucination",
                flags=["no_context"],
            )

        best_score = max(
            (r.rrf_score if hasattr(r, "rrf_score") else r.get("rrf_score", 0))
            for r in retrieval_results
        )
        if best_score < threshold:
            return GuardrailResult(
                action=GuardrailAction.WARN,
                reason=f"Low relevance ({best_score:.3f} < {threshold}) — answer may not be grounded",
                flags=["low_relevance"],
            )

        return GuardrailResult(action=GuardrailAction.ALLOW, reason="Context is relevant")
