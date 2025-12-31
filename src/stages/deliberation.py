import os
from typing import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.state import Runnable
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.conversation.conversation_state import (
    ConversationStage,
    ConversationState,
    serialize_messages,
)
from src.events import EventType
from src.prompts import get_deliberation_prompt
from src.utils.events import stream_single_message
from src.utils.messages import chunk_to_text
from src.services.firestore_service import update_conversation

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL") or "openai/gpt-5.1",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
)


def start_deliberation(state: ConversationState):
    active_listening_summary, party_positioning_summary, perspective_taking_summary = (
        get_required_summaries(state)
    )
    deliberation_prompt = get_deliberation_prompt(
        state.topic,
        active_listening_summary,
        party_positioning_summary,
        perspective_taking_summary,
    )

    deliberation_agent: Runnable = create_agent(
        model=llm,
        system_prompt=str(deliberation_prompt),
    )

    state.deliberation_messages = []
    update_conversation(
        conversation_id=state.id,
        extra={
            "deliberation_messages": serialize_messages(state.deliberation_messages)
        },
    )
    return stream_response_and_update_state(state, deliberation_agent)


def deliberation(state: ConversationState, user_message: str) -> Iterator[dict]:
    active_listening_summary, party_positioning_summary, perspective_taking_summary = (
        get_required_summaries(state)
    )
    deliberation_prompt = get_deliberation_prompt(
        state.topic,
        active_listening_summary,
        party_positioning_summary,
        perspective_taking_summary,
    )

    state.deliberation_messages.append(HumanMessage(content=user_message))

    @tool
    def end_deliberation(deliberation_summary: str) -> str:
        """This tool is used to end the deliberation phase, populate the parameters with the user's reflections, updated perspective and defending arguments (approved by the user) to finish this conversation phase phase.
        Formulate the the summary in 3rd person form (the user...)"""
        state.stage = ConversationStage.PARTY_MATCHING
        state.deliberation_summary = deliberation_summary
        update_conversation(
            conversation_id=state.id,
            stage=ConversationStage.PARTY_MATCHING.value,
            extra={"deliberation_summary": deliberation_summary},
        )
        return "Deliberation phase completed"

    deliberation_agent: Runnable = create_agent(
        model=llm,
        tools=[end_deliberation],
        system_prompt=deliberation_prompt,
    )

    return stream_response_and_update_state(state, deliberation_agent)


def get_required_summaries(state: ConversationState) -> tuple[str, str, str]:
    party_positioning_summary = state.party_positioning_summary
    if party_positioning_summary is None:
        raise ValueError(
            "Party positioning summary is required to start the perspective taking phase"
        )

    active_listening_summary = state.active_listening_summary
    if active_listening_summary is None:
        raise ValueError(
            "Active listening summary is required to start the perspective taking phase"
        )

    perspective_taking_summary = state.perspective_taking_summary
    if perspective_taking_summary is None:
        raise ValueError(
            "Perspective taking summary is required to start the deliberation phase"
        )

    return (
        active_listening_summary,
        party_positioning_summary,
        perspective_taking_summary,
    )


def stream_response_and_update_state(
    state: ConversationState, agent: Runnable
) -> Iterator[dict]:
    assistant_message_text = ""
    tool_called = False
    stream_open = False

    for chunk, metadata in agent.stream(
        {"messages": state.deliberation_messages}, stream_mode="messages"
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
        state.deliberation_messages.append(AIMessage(content=assistant_message_text))

    if stream_open and not tool_called:
        yield {"type": EventType.MESSAGE_END.value}

    update_conversation(
        conversation_id=state.id,
        extra={
            "deliberation_messages": serialize_messages(state.deliberation_messages)
        },
    )

    if tool_called:
        print("Ending deliberation phase, starting party matching")
        from src.stages.party_matching import start_party_matching

        yield from start_party_matching(state)
