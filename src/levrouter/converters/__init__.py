"""Feature extraction converters for different LLM frameworks."""

from __future__ import annotations

from typing import Callable


def get_extractor(framework: str) -> Callable:
    """Get the feature extraction function for a framework.

    Args:
        framework: One of "langchain", "anthropic", "openai"

    Returns:
        A callable that extracts RequestFeatures from framework-specific messages

    Raises:
        ValueError: If framework is unknown
    """
    if framework == "langchain":
        from .langchain import extract
        return extract
    elif framework == "anthropic":
        from .anthropic import extract
        return extract
    elif framework == "openai":
        from .openai import extract
        return extract
    else:
        raise ValueError(f"Unknown framework: {framework}. Choose from: langchain, anthropic, openai")
