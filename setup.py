"""Make the voice_rag package installable in development mode."""
from setuptools import setup, find_packages

setup(
    name="voice-rag",
    version="1.0.0",
    description="Voice-Enabled RAG over MSMARCO-XI",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
)
