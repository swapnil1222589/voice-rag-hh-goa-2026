"""Orchestration harness — structured pipeline with retries, error recovery,
structured I/O, and tool-call-style stage management.

The harness wraps the entire pipeline (STT → guardrail → retrieve → generate →
guardrail) with:
  - Structured input/output (Pydantic models for every stage)
  - Per-stage retries with tenacity
  - Error recovery (graceful degradation at each stage)
  - Latency tracking per stage
  - Decision logging (why the system did what it did)
  - A tool-call interface so the pipeline can be invoked programmatically

This is NOT a single prompt-in/text-out call — it's a multi-stage orchestration
graph with explicit error handling at each node.
"""
from .pipeline import run_pipeline, run_text_pipeline, PipelineResult, PipelineContext

__all__ = ["run_pipeline", "run_text_pipeline", "PipelineResult", "PipelineContext"]
