"""Answer generation — OpenAI GPT-class model with grounded context.

The generator takes the user's query and retrieved passages, constructs a
prompt that instructs the model to answer *only* from the retrieved context,
and returns a structured result with the answer and grounding info.
"""
from .generator import AnswerGenerator, GenerationResult

__all__ = ["AnswerGenerator", "GenerationResult"]
