"""Utilities for handling LangChain message chunks."""

from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage


def chunk_to_text(chunk: object) -> str:
    """Convert streamed chunks (strings, messages, dicts) into plain text."""
    if chunk is None:
        return ""

    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, BaseMessage):
        content = chunk.content
    else:
        content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
        return "".join(parts)

    return str(content)
