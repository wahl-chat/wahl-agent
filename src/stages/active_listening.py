from typing import Iterator
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from pydantic import SecretStr
from src.conversation.conversation_state import (
    ConversationState,
    ConversationStage,
    serialize_messages,
)
from src.prompts import get_active_listening_prompt
from src.events import EventType
from src.stages.party_positioning import start_party_positioning
from src.utils.messages import chunk_to_text
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import os

from src.services.firestore_service import update_conversation

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL") or "openai/gpt-5.1",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
)


def active_listening(state: ConversationState, user_message: str) -> Iterator[dict]:
    state.active_listening_messages.append(HumanMessage(content=user_message))

    @tool
    def end_active_listening(user_perspective_summary: str) -> str:
        """This tool is used to end the active listening phase and populate the parameters with the user's perspectives summary (approved by the user)to finish the active listening phase.
        Formulate the user's perspective summary in 3rd person form"""
        state.stage = ConversationStage.PARTY_POSITIONING
        state.active_listening_summary = user_perspective_summary
        update_conversation(
            conversation_id=state.id,
            stage=ConversationStage.PARTY_POSITIONING.value,
            extra={"active_listening_summary": user_perspective_summary},
        )
        return "Active listening phase completed"

    active_listening_agent: Runnable = create_agent(
        model=llm,
        tools=[end_active_listening],
        system_prompt=str(get_active_listening_prompt(state.topic).content),
    )

    assistant_message_text = ""
    tool_called = False
    stream_open = False

    for chunk, metadata in active_listening_agent.stream(
        {"messages": state.active_listening_messages}, stream_mode="messages"
    ):
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
        state.active_listening_messages.append(
            AIMessage(content=assistant_message_text)
        )

    if stream_open and not tool_called:
        yield {"type": EventType.MESSAGE_END.value}

    update_conversation(
        conversation_id=state.id,
        extra={
            "active_listening_messages": serialize_messages(
                state.active_listening_messages
            )
        },
    )

    if tool_called:
        yield from start_party_positioning(state)
