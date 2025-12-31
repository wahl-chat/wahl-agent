from enum import Enum
from typing import Iterable, Sequence

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
)


class ConversationStage(Enum):
    START = "start"
    ACTIVE_LISTENING = "active_listening"
    PARTY_POSITIONING = "party_positioning"
    PERSPECTIVE_TAKING = "perspective_taking"
    DELIBERATION = "deliberation"
    PARTY_MATCHING = "party_matching"
    END = "end"


class ConversationState:
    def __init__(
        self,
        topic: str,
        id: str,
        stage: ConversationStage = ConversationStage.START,
        active_listening_messages: Sequence[BaseMessage] | None = None,
        party_positioning_messages: Sequence[BaseMessage] | None = None,
        perspective_taking_messages: Sequence[BaseMessage] | None = None,
        deliberation_messages: Sequence[BaseMessage] | None = None,
        active_listening_summary: str | None = None,
        party_positioning_summary: str | None = None,
        perspective_taking_summary: str | None = None,
        deliberation_summary: str | None = None,
    ):
        self.id: str = id
        self.stage: ConversationStage = stage
        self.topic: str = topic
        self.active_listening_messages: list[BaseMessage] = (
            list(active_listening_messages) if active_listening_messages else []
        )
        self.party_positioning_messages: list[BaseMessage] = (
            list(party_positioning_messages) if party_positioning_messages else []
        )
        self.perspective_taking_messages: list[BaseMessage] = (
            list(perspective_taking_messages) if perspective_taking_messages else []
        )
        self.deliberation_messages: list[BaseMessage] = (
            list(deliberation_messages) if deliberation_messages else []
        )
        self.active_listening_summary: str | None = active_listening_summary
        self.party_positioning_summary: str | None = party_positioning_summary
        self.perspective_taking_summary: str | None = perspective_taking_summary
        self.deliberation_summary: str | None = deliberation_summary


MESSAGE_TYPE_TO_CLASS = {
    "ai": AIMessage,
    "human": HumanMessage,
    "system": SystemMessage,
    "function": FunctionMessage,
    "tool": ToolMessage,
}


def serialize_message(message: BaseMessage) -> dict:
    return {
        "type": message.type,
        "content": message.content,
        "additional_kwargs": message.additional_kwargs,
        "response_metadata": getattr(message, "response_metadata", None),
    }


def serialize_messages(messages: Iterable[BaseMessage]) -> list[dict]:
    return [serialize_message(msg) for msg in messages]


def deserialize_message(payload: dict | str | BaseMessage) -> BaseMessage:
    if isinstance(payload, BaseMessage):
        return payload
    if isinstance(payload, str):
        return HumanMessage(content=payload)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported message payload type: {type(payload)}")

    msg_type = payload.get("type", "human")
    content = payload.get("content", "")
    additional_kwargs = payload.get("additional_kwargs", {}) or {}
    response_metadata = payload.get("response_metadata", {}) or {}

    message_cls = MESSAGE_TYPE_TO_CLASS.get(msg_type, HumanMessage)
    message = message_cls(
        content=content,
        additional_kwargs=additional_kwargs,
    )
    if hasattr(message, "response_metadata") and response_metadata:
        message.response_metadata = response_metadata
    return message


def deserialize_messages(
    payloads: Iterable[dict | str | BaseMessage],
) -> list[BaseMessage]:
    return [deserialize_message(payload) for payload in payloads]
