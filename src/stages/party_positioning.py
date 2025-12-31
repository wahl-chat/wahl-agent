from typing import Iterator, Any
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
from src.events import EventType
from src.stages.perspective_taking import start_perspective_taking
from src.utils.messages import chunk_to_text
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.prompts import get_party_positioning_prompt
import os

from src.services.firestore_service import (
    update_conversation,
    get_party_positions_by_topic_id,
)

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL") or "openai/gpt-5.1",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
)


def start_party_positioning(state: ConversationState) -> Iterator[dict]:
    active_listening_summary = state.active_listening_summary
    if active_listening_summary is None:
        raise ValueError(
            "User perspective summary is required to start the party positioning phase"
        )
    party_positions = get_party_positions(state.topic)
    party_positioning_prompt = get_party_positioning_prompt(
        topic=state.topic,
        party_positions=party_positions,
        active_listening_summary=active_listening_summary,
    )

    party_positioning_agent: Runnable = create_agent(
        model=llm,
        tools=[],
        system_prompt=str(party_positioning_prompt.content),
    )

    state.party_positioning_messages = []
    update_conversation(
        conversation_id=state.id,
        extra={
            "party_positioning_messages": serialize_messages(
                state.party_positioning_messages
            )
        },
    )
    return stream_response_and_update_state(state, party_positioning_agent)


def party_positioning(state: ConversationState, user_message: str) -> Iterator[dict]:
    active_listening_summary = state.active_listening_summary
    if active_listening_summary is None:
        raise ValueError(
            "User perspective summary is required to start the party positioning phase"
        )
    party_positions = get_party_positions(state.topic)
    party_positioning_prompt = get_party_positioning_prompt(
        topic=state.topic,
        party_positions=party_positions,
        active_listening_summary=active_listening_summary,
    )

    state.party_positioning_messages.append(HumanMessage(content=user_message))

    @tool
    def end_party_positioning(user_desired_goal_and_methods: str) -> str:
        """This tool is used to end the party positioning phase, populate the parameters with the user's desired goals and method (approved by the user)to finish this conversation phase phase.
        Formulate the summary in 3rd person form (the user...)"""
        state.stage = ConversationStage.PERSPECTIVE_TAKING
        state.party_positioning_summary = user_desired_goal_and_methods
        update_conversation(
            conversation_id=state.id,
            stage=ConversationStage.PERSPECTIVE_TAKING.value,
            extra={"party_positioning_summary": user_desired_goal_and_methods},
        )
        return "Party positioning phase completed"

    party_positioning_agent: Runnable = create_agent(
        model=llm,
        tools=[end_party_positioning],
        system_prompt=str(party_positioning_prompt.content),
    )

    return stream_response_and_update_state(state, party_positioning_agent)


def get_party_positions(topic: str) -> list[tuple[str, dict[str, Any]]]:
    topic_id = get_topic_id(topic)
    party_positions = get_party_positions_by_topic_id(topic_id)
    if party_positions is None:
        raise ValueError(f"No party positions found for topic {topic}")
    return party_positions


def get_topic_id(topic: str) -> str:
    if topic == "Migration":
        return "migration_security_state"
    elif topic == "Wirtschaft":
        return "economy_work_social"
    elif topic == "Umwelt und Klima":
        return "energy_climate_environment"
    else:
        raise ValueError(f"Invalid topic: {topic}")


def stream_response_and_update_state(
    state: ConversationState, agent: Runnable
) -> Iterator[dict]:
    assistant_message_text = ""
    tool_called = False
    stream_open = False

    for chunk, metadata in agent.stream(
        {"messages": state.party_positioning_messages}, stream_mode="messages"
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
        state.party_positioning_messages.append(
            AIMessage(content=assistant_message_text)
        )

    if stream_open and not tool_called:
        yield {"type": EventType.MESSAGE_END.value}

    update_conversation(
        conversation_id=state.id,
        extra={
            "party_positioning_messages": serialize_messages(
                state.party_positioning_messages
            )
        },
    )

    if tool_called:
        print("Starting perspective taking phase")
        yield from start_perspective_taking(state)
