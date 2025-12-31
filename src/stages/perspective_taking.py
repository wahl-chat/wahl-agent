import os
from typing import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import Runnable
from pydantic import SecretStr

from dotenv import load_dotenv

from src.conversation.conversation_state import (
    ConversationStage,
    ConversationState,
    serialize_messages,
)
from src.events import EventType
from src.prompts import get_perspective_taking_prompt
from src.stages.deliberation import start_deliberation
from src.utils.messages import chunk_to_text
from src.services.firestore_service import update_conversation

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL") or "openai/gpt-5.1",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
)

perplexity_client = ChatOpenAI(
    model="perplexity/sonar-pro",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
)


@tool
def perplexity_search(query: str) -> str:
    """Search the Internet for the query using Perplexity search."""
    response = perplexity_client.invoke([HumanMessage(content=query)])
    return str(response.content)


def start_perspective_taking(state: ConversationState) -> Iterator[dict]:
    active_listening_summary, party_positioning_summary = get_required_summaries(state)

    perspective_taking_prompt = get_perspective_taking_prompt(
        topic=state.topic,
        active_listening_summary=active_listening_summary,
        party_positioning_summary=party_positioning_summary,
    )

    perspective_taking_agent: Runnable = create_agent(
        model=llm,
        tools=[perplexity_search],
        system_prompt=str(perspective_taking_prompt),
    )

    state.perspective_taking_messages = []
    update_conversation(
        conversation_id=state.id,
        extra={
            "perspective_taking_messages": serialize_messages(
                state.perspective_taking_messages
            )
        },
    )
    return stream_response_and_update_state(state, perspective_taking_agent)


def perspective_taking(state: ConversationState, user_message: str) -> Iterator[dict]:
    active_listening_summary, party_positioning_summary = get_required_summaries(state)

    perspective_taking_prompt = get_perspective_taking_prompt(
        topic=state.topic,
        active_listening_summary=active_listening_summary,
        party_positioning_summary=party_positioning_summary,
    )

    @tool
    def end_perspective_taking(perspective_taking_summary: str) -> str:
        """This tool is used to end the perspective taking phase, populate the parameters with the user's thoughts and feelings on the the situation
        proposed by the perspective taking exercise (approved by the user) to finish this conversation phase. Don't include user information already provided in the previous summaries.
        Formulate the user's goals in 3rd person form (the user...)"""
        return end_perspective_taking_callback(perspective_taking_summary, state)

    perspective_taking_agent: Runnable = create_agent(
        model=llm,
        tools=[perplexity_search, end_perspective_taking],
        system_prompt=str(perspective_taking_prompt),
    )

    state.perspective_taking_messages.append(HumanMessage(content=user_message))

    return stream_response_and_update_state(state, perspective_taking_agent)


def end_perspective_taking_callback(
    perspective_taking_summary: str, state: ConversationState
) -> str:
    """This tool is used to end the perspective taking phase in the, populate the parameters with the user's desired goals and method (approved by the user)to finish this conversation phase phase.
    Formulate the user's goals in 3rd person form (the user...)"""
    state.stage = ConversationStage.DELIBERATION
    state.perspective_taking_summary = perspective_taking_summary
    update_conversation(
        conversation_id=state.id,
        stage=ConversationStage.DELIBERATION.value,
        extra={"perspective_taking_summary": perspective_taking_summary},
    )
    return "Party positioning phase completed"


def get_required_summaries(state: ConversationState) -> tuple[str, str]:
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

    return active_listening_summary, party_positioning_summary


def stream_response_and_update_state(state: ConversationState, agent: Runnable):
    assistant_message_text = ""
    end_tool_called = False
    stream_open = False

    for chunk, metadata in agent.stream(
        {"messages": state.perspective_taking_messages}, stream_mode="messages"
    ):
        chunk_type = getattr(chunk, "type", None)
        if isinstance(chunk, BaseMessage):
            chunk_type = chunk.type

        if chunk_type == "tool":
            tool_name = getattr(chunk, "name", None)
            if tool_name == "end_perspective_taking":
                end_tool_called = True
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

    if assistant_message_text and not end_tool_called:
        state.perspective_taking_messages.append(
            AIMessage(content=assistant_message_text)
        )

    if stream_open and not end_tool_called:
        yield {"type": EventType.MESSAGE_END.value}

    update_conversation(
        conversation_id=state.id,
        extra={
            f"{state.stage.value}_messages": serialize_messages(
                state.perspective_taking_messages
            )
        },
    )

    if end_tool_called:
        print("Starting deliberation phase")
        yield from start_deliberation(state)
