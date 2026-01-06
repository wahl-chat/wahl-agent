from __future__ import annotations

from typing import Iterable, Iterator

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.state import Runnable

from src.conversation.conversation_state import (
    ConversationState,
    serialize_messages,
)
from src.events import EventType
from src.utils.messages import chunk_to_text
from src.services.firestore_service import update_conversation


def stream_text_as_events(text_chunks: Iterable[str]) -> Iterator[dict]:
    """Convert one or more text chunks into the standard SSE event sequence."""
    iterator = iter(text_chunks)
    try:
        first_chunk = next(iterator)
    except StopIteration:
        yield {"type": EventType.END.value}
        return

    yield {"type": EventType.MESSAGE_START.value}
    yield {
        "type": EventType.MESSAGE_CHUNK.value,
        "content": first_chunk,
    }

    for chunk in iterator:
        yield {
            "type": EventType.MESSAGE_CHUNK.value,
            "content": chunk,
        }

    yield {"type": EventType.MESSAGE_END.value}
    yield {"type": EventType.END.value}


def stream_single_message(text: str) -> Iterator[dict]:
    yield from stream_text_as_events([text])


def progress_event(message: str) -> dict:
    """Create a progress update event to inform the user about ongoing operations."""
    return {"type": EventType.PROGRESS_UPDATE.value, "content": message}


def stream_response_and_update_state(
    state: ConversationState,
    agent: Runnable,
    messages: list[BaseMessage],
    next_iterator: Iterator[dict],
) -> Iterator[dict]:
    assistant_message_text = ""
    tool_called = False
    stream_open = False

    for chunk, metadata in agent.stream({"messages": messages}, stream_mode="messages"):
        chunk_type = getattr(chunk, "type", None)
        if isinstance(chunk, BaseMessage):
            chunk_type = chunk.type

        if chunk_type == "tool":
            tool_called = True
            break

        chunk_text = chunk_to_text(chunk)
        if chunk_text:
            if not stream_open:
                yield {"type": EventType.MESSAGE_START.value}
                stream_open = True
            assistant_message_text += chunk_text
            yield {
                "type": EventType.MESSAGE_CHUNK.value,
                "content": chunk_text,
            }

    if assistant_message_text and not tool_called:
        messages.append(AIMessage(content=assistant_message_text))

    if stream_open and not tool_called:
        yield {"type": EventType.MESSAGE_END.value}
    yield {"type": EventType.END.value}

    update_conversation(
        conversation_id=state.id,
        extra={f"{state.stage.value}_messages": serialize_messages(messages)},
    )

    if tool_called:
        yield from next_iterator
