"""Output guardrail — hallucination and grounding checks.

After the LLM generates an answer, we verify it's grounded in the retrieved
context.  Three checks:

1. **Claim extraction & grounding**: Extract factual claims (numbers, names, dates)
   from the answer and check each appears in the context.  Uses a lightweight
   regex-based extractor — no external API call needed.

2. **Refusal propagation**: If the LLM's answer is a refusal ("I don't have enough
   information"), that's actually a GOOD outcome — the system correctly refused.

3. **Answer relevance**: Check that the answer is actually about the question,
   not a tangent.  Uses token overlap between the answer and the query.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Refusal patterns — the LLM correctly refusing to answer
_REFUSAL_PATTERNS = [
    r"i don't have enough information",
    r"i do not have enough information",
    r"the (provided )?context (does not|doesn't) (contain|include|mention|provide)",
    r"not (enough |sufficient )?(information|context) (in|to|for)",
    r"cannot (answer|provide) (based on|from) the (provided )?context",
    r"based on the (provided )?context,? i (cannot|can't|cannot) ",
    r"no (relevant )?(information|context|passage)",
]

_COMPILED_REFUSAL = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS]

# Factual claim extractors
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?(?:,\d{3})*(?:\s?%)?\b')
_DATE_RE = re.compile(
    r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}|'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4})\b',
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r'\b(?:19|20)\d{2}\b')

# Words to ignore for overlap (common, not informative)
_STOP = set("the a an is are was were be been being have has had do does did "
            "will would could should may might can to of in on at for with by "
            "from as it its this that these those i you he she we they my your "
            "what which who when where why how not no yes and or but if so".split())


@dataclass
class HallucinationCheckResult:
    is_grounded: bool
    confidence: float          # 0-1, how confident we are the answer is grounded
    ungrounded_claims: list = field(default_factory=list)  # claims not found in context
    is_refusal: bool = False   # the LLM correctly refused
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "is_grounded": self.is_grounded,
            "confidence": self.confidence,
            "ungrounded_claims": self.ungrounded_claims,
            "is_refusal": self.is_refusal,
            "reason": self.reason,
        }


class OutputGuardrail:
    """Checks the generated answer for hallucination and grounding."""

    def check(
        self,
        answer: str,
        context_texts: List[str],
        query: str = "",
    ) -> HallucinationCheckResult:
        """Run grounding checks on the answer.

        Args:
            answer: The generated answer text.
            context_texts: List of retrieved passage texts.
            query: The original query (for relevance check).
        """
        if not answer or not answer.strip():
            return HallucinationCheckResult(
                is_grounded=False,
                confidence=0.0,
                reason="Empty answer",
            )

        # 1. Check if this is a refusal — that's a good outcome
        is_refusal = any(p.search(answer) for p in _COMPILED_REFUSAL)
        if is_refusal:
            return HallucinationCheckResult(
                is_grounded=True,
                confidence=1.0,
                is_refusal=True,
                reason="LLM correctly refused — insufficient context",
            )

        if not context_texts:
            return HallucinationCheckResult(
                is_grounded=False,
                confidence=0.0,
                ungrounded_claims=["no_context"],
                reason="No context provided to ground the answer",
            )

        combined_context = " ".join(context_texts).lower()

        # 2. Extract factual claims and check grounding
        claims = self._extract_claims(answer)
        ungrounded = []
        for claim in claims:
            if claim.lower() not in combined_context:
                ungrounded.append(claim)

        # 3. Answer-query relevance (token overlap)
        relevance = self._relevance_score(answer, query) if query else 0.5

        # 4. Compute confidence
        total_claims = len(claims) if claims else 1
        grounded_ratio = 1.0 - (len(ungrounded) / total_claims)
        confidence = 0.6 * grounded_ratio + 0.4 * relevance

        is_grounded = len(ungrounded) == 0 and confidence >= 0.4

        reason = ""
        if ungrounded:
            reason = f"Found {len(ungrounded)} ungrounded claims: {ungrounded[:3]}"
        elif confidence < 0.4:
            reason = f"Low confidence ({confidence:.2f}) — answer may not be relevant to query"
        else:
            reason = "All claims grounded in context"

        return HallucinationCheckResult(
            is_grounded=is_grounded,
            confidence=round(confidence, 3),
            ungrounded_claims=ungrounded,
            reason=reason,
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims (numbers, dates, years) from the answer."""
        claims: List[str] = []
        claims.extend(m.group() for m in _NUMBER_RE.finditer(text))
        claims.extend(m.group() for m in _DATE_RE.finditer(text))
        claims.extend(m.group() for m in _YEAR_RE.finditer(text))
        return list(set(claims))  # dedupe

    def _relevance_score(self, answer: str, query: str) -> float:
        """Token overlap between query and answer (Jaccard on content words)."""
        q_words = {w.lower() for w in query.split() if w.lower() not in _STOP and len(w) > 2}
        a_words = {w.lower() for w in answer.split() if w.lower() not in _STOP and len(w) > 2}
        if not q_words or not a_words:
            return 0.0
        return len(q_words & a_words) / len(q_words | a_words)
