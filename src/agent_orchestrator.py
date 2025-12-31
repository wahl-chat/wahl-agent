from typing import Iterator
from openai import OpenAI
import os
from dotenv import load_dotenv

from src.stages.deliberation import deliberation
from src.utils.events import stream_single_message

from src.conversation.conversation_state import (
    ConversationState,
    ConversationStage,
    deserialize_messages,
)
from src.stages.active_listening import active_listening
from src.stages.start import start
from src.stages.party_positioning import party_positioning
from src.stages.perspective_taking import perspective_taking

from src.services.firestore_service import get_conversation

load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


def chat(conversation_id: str, user_message: str) -> Iterator[dict]:
    conversation = get_conversation_by_id(conversation_id)
    stage = conversation.stage
    match stage:
        case ConversationStage.START:
            print("Starting conversation")
            return start(conversation, user_message)
        case ConversationStage.ACTIVE_LISTENING:
            print("Conversation in active listening stage")
            return active_listening(conversation, user_message)
        case ConversationStage.PARTY_POSITIONING:
            print("Conversation in party positioning stage")
            return party_positioning(conversation, user_message)
        case ConversationStage.PERSPECTIVE_TAKING:
            print("Conversation in perspective taking stage")
            return perspective_taking(conversation, user_message)
        case ConversationStage.DELIBERATION:
            print("Conversation in deliberation stage")
            return deliberation(conversation, user_message)
        case ConversationStage.PARTY_MATCHING:
            print("Conversation in party matching stage")
            return stream_single_message("Implement party matching stage")
        case ConversationStage.END:
            print("Reached End tage")
            return stream_single_message("Dialog ist beendet. Danke für die Teilnahme.")


def get_conversation_by_id(conversation_id: str) -> ConversationState:
    firestore_doc = get_conversation(conversation_id)

    if not firestore_doc:
        raise ValueError(
            f"Conversation with ID '{conversation_id}' not found in Firestore"
        )

    # Create ConversationState with topic from Firestore
    conversation_state = ConversationState(
        topic=firestore_doc["topic"],
        id=conversation_id,
        active_listening_messages=deserialize_messages(
            firestore_doc.get("active_listening_messages", [])
        ),
        party_positioning_messages=deserialize_messages(
            firestore_doc.get("party_positioning_messages", [])
        ),
        perspective_taking_messages=deserialize_messages(
            firestore_doc.get("perspective_taking_messages", [])
        ),
        deliberation_messages=deserialize_messages(
            firestore_doc.get("deliberation_messages", [])
        ),
        active_listening_summary=firestore_doc.get("active_listening_summary", None),
        party_positioning_summary=firestore_doc.get("party_positioning_summary", None),
        perspective_taking_summary=firestore_doc.get(
            "perspective_taking_summary", None
        ),
        deliberation_summary=firestore_doc.get("deliberation_summary", None),
    )

    # Convert stage string to ConversationStage enum
    stage_str = firestore_doc.get("stage", "start")
    try:
        conversation_state.stage = ConversationStage(stage_str)
    except ValueError:
        # If stage doesn't match any enum value, default to START
        conversation_state.stage = ConversationStage.START

    return conversation_state
