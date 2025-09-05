"""Tooling entrypoints for recommendation and retrieval utilities.

Currently exposes `ToolExecutor`, which aggregates SQL, BM25, embedding, and
semantic-ID tools behind a single interface for the agent to call.
"""
from tpa.environments.tools.executor import ToolExecutor
__all__ = ["ToolExecutor"]
