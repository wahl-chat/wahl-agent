import os
from langchain_core.prompts import ChatPromptTemplate
from typing import Iterator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from src.conversation.conversation_state import (
    ConversationStage,
    ConversationState,
)
from src.prompts import (
    get_distillation_prompt,
    get_party_matching_prompt,
)
from src.utils.events import stream_single_message
from src.services.firestore_service import update_conversation
from src.services.wahl_chat_service import (
    WahlChatResponse,
    ask_bundestag_parties,
)


load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL") or "openai/gpt-5.1",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
)


def start_party_matching(state: ConversationState) -> Iterator[dict]:
    deliberation_summary = get_required_summaries(state)

    party_question_distillation_prompt = ChatPromptTemplate.from_template(
        get_distillation_prompt()
    )

    question_distillation_chain = (
        party_question_distillation_prompt | llm | StrOutputParser()
    )

    question = question_distillation_chain.invoke(
        {"topic": state.topic, "deliberation_summary": deliberation_summary}
    )

    party_responses: WahlChatResponse = ask_bundestag_parties(question)
    party_matching_prompt = ChatPromptTemplate.from_template(
        get_party_matching_prompt(party_responses)
    )

    party_matching_chain = party_matching_prompt | llm | StrOutputParser()

    party_matching_result = party_matching_chain.invoke(
        {
            "topic": state.topic,
            "deliberation_summary": deliberation_summary,
            "question": question,
        }
    )

    yield from stream_single_message(party_matching_result)

    update_conversation(
        conversation_id=state.id,
        stage=ConversationStage.END.value,
        extra={"party_matching_result": party_matching_result},
    )


def get_required_summaries(state: ConversationState) -> str:
    deliberation_summary = state.deliberation_summary
    if deliberation_summary is None:
        raise ValueError(
            "Deliberation summary is required to start the party matching phase"
        )

    return deliberation_summary
