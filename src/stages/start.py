from src.conversation.conversation_state import (
    ConversationState,
    ConversationStage,
    serialize_messages,
)
from src.prompts import get_initial_message
from src.utils.events import stream_single_message
from langchain_core.messages import AIMessage
from src.services.firestore_service import update_conversation
from typing import Iterator


def start(state: ConversationState, user_message: str) -> Iterator[dict]:
    initial_message: AIMessage = get_initial_message(state.topic)
    state.active_listening_messages.append(initial_message)
    state.stage = ConversationStage.ACTIVE_LISTENING

    update_conversation(
        conversation_id=state.id,
        stage=ConversationStage.ACTIVE_LISTENING.value,
        extra={
            "active_listening_messages": serialize_messages(
                state.active_listening_messages
            )
        },
    )

    return stream_single_message(str(initial_message.content))
