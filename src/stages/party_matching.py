import os
from datetime import datetime, timezone
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
from src.utils.events import stream_single_message, progress_event, sources_ready_event
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
    yield progress_event("Deine Diskussion wird zusammengefasst")
    deliberation_summary = get_required_summaries(state)

    party_question_distillation_prompt = ChatPromptTemplate.from_template(
        get_distillation_prompt()
    )

    question_distillation_chain = (
        party_question_distillation_prompt | llm | StrOutputParser()
    )

    yield progress_event("Kernfrage wird formuliert")
    question = question_distillation_chain.invoke(
        {"topic": state.topic, "deliberation_summary": deliberation_summary}
    )
    print("Question asked to parties: ", question)

    yield progress_event("Parteipositionen werden abgefragt")
    party_responses: WahlChatResponse = ask_bundestag_parties(question)

    # Build sources payload for frontend (per-party grouped)
    sources_payload = [
        {
            "party_id": party_response.party_id,
            "sources": [
                {
                    "source": s.name,
                    "page": s.page,
                    "url": s.url or "",
                    "document_publish_date": s.document_publish_date or "",
                    "source_document": s.source_document or "",
                }
                for s in party_response.sources
            ],
        }
        for party_response in party_responses.party_responses
        if party_response.sources
    ]

    party_matching_prompt = ChatPromptTemplate.from_template(
        get_party_matching_prompt(party_responses)
    )

    print("Party matching prompt: ", party_matching_prompt)

    party_matching_chain = party_matching_prompt | llm | StrOutputParser()

    yield progress_event("Übereinstimmung wird analysiert")
    party_matching_result = party_matching_chain.invoke(
        {
            "topic": state.topic,
            "deliberation_summary": deliberation_summary,
            "question": question,
        }
    )

    # Emit sources before the message content
    if sources_payload:
        yield sources_ready_event(sources_payload)

    yield from stream_single_message(party_matching_result)

    update_conversation(
        conversation_id=state.id,
        stage=ConversationStage.END.value,
        extra={
            "party_matching_result": party_matching_result,
            "party_matching_sources": sources_payload,
            "ended_at": datetime.now(timezone.utc),
        },
    )


def get_required_summaries(state: ConversationState) -> str:
    deliberation_summary = state.deliberation_summary
    if deliberation_summary is None:
        raise ValueError(
            "Deliberation summary is required to start the party matching phase"
        )

    return deliberation_summary
