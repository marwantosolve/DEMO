"""
Adapters for different MAS frameworks.

Provides plug-and-play integration with Google ADK, LangGraph, AutoGen, etc.
"""

from mas_eval.adapters.adk_adapter import ADKAdapter, ADKTracingCallback

__all__ = ["ADKAdapter", "ADKTracingCallback"]
