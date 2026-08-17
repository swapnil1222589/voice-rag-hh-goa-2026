"""Guardrails — input validation, off-topic detection, hallucination checking.

Guardrail layers:
1. **Input guardrail** — validates the transcribed query:
   - Empty / too short / too long
   - Unsafe or inappropriate content (profanity, violence, self-harm, PII)
   - Off-topic (not related to the passage corpus — detected via low retrieval relevance)
2. **Output guardrail** — checks the generated answer:
   - Hallucination check: does the answer contain claims not grounded in the context?
   - Refusal propagation: if context is insufficient, the system should say so
   - Confidence calibration: low retrieval scores → refuse

The system knows when NOT to answer — that's the whole point.
"""
from .input_guard import InputGuardrail, GuardrailResult, GuardrailAction
from .output_guard import OutputGuardrail, HallucinationCheckResult

__all__ = [
    "InputGuardrail", "GuardrailResult", "GuardrailAction",
    "OutputGuardrail", "HallucinationCheckResult",
]
