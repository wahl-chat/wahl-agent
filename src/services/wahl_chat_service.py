import asyncio
import os
import uuid
from dataclasses import dataclass, field

import socketio
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv(
    "WAHL_CHAT_BACKEND_URL",
    "https://wahl-chat-backend-dev-670868139461.europe-west1.run.app",
)
CONTEXT_ID = os.getenv("WAHL_CHAT_CONTEXT_ID", "bundestagswahl-2025")
PARTY_IDS = ["spd", "cdu", "gruene", "afd", "linke"]
TIMEOUT_SECONDS = 60


@dataclass
class Source:
    name: str
    page: int
    url: str | None = None
    document_publish_date: str | None = None
    source_document: str | None = None


@dataclass
class PartyResponse:
    party_id: str
    response: str
    sources: list[Source] = field(default_factory=list)


@dataclass
class WahlChatResponse:
    party_responses: list[PartyResponse]


async def _ask_parties_async(question: str) -> WahlChatResponse:
    sio = socketio.AsyncClient()
    session_id = str(uuid.uuid4())

    responses: dict[str, str] = {}
    sources: dict[str, list[Source]] = {}
    complete_event = asyncio.Event()

    @sio.on("chat_session_initialized")
    async def on_initialized(data):
        await sio.emit(
            "chat_answer_request",
            {
                "session_id": session_id,
                "context_id": CONTEXT_ID,
                "user_message": question,
                "party_ids": PARTY_IDS,
                "user_is_logged_in": False,
            },
        )

    @sio.on("sources_ready")
    async def on_sources(data):
        party_id = data["party_id"]
        sources[party_id] = [
            Source(
                name=s.get("source", ""),
                page=s.get("page", 0),
                url=s.get("url"),
                document_publish_date=s.get("document_publish_date"),
                source_document=s.get("source_document"),
            )
            for s in data.get("sources", [])
        ]

    @sio.on("party_response_complete")
    async def on_party_complete(data):
        responses[data["party_id"]] = data["complete_message"]

    @sio.on("chat_response_complete")
    async def on_complete(data):
        complete_event.set()

    await sio.connect(
        BACKEND_URL,
        transports=["websocket"],
        headers={"Origin": "http://localhost:3000"},
    )

    await sio.emit(
        "chat_session_init",
        {
            "session_id": session_id,
            "context_id": CONTEXT_ID,
            "party_ids": PARTY_IDS,
            "chat_history": [],
            "current_title": "",
            "chat_response_llm_size": "large",
            "last_quick_replies": [],
            "is_cacheable": True,
        },
    )

    await asyncio.wait_for(complete_event.wait(), timeout=TIMEOUT_SECONDS)
    await sio.disconnect()

    return WahlChatResponse(
        party_responses=[
            PartyResponse(
                party_id=party_id,
                response=responses.get(party_id, ""),
                sources=sources.get(party_id, []),
            )
            for party_id in PARTY_IDS
        ]
    )


def ask_bundestag_parties(question: str) -> WahlChatResponse:
    return asyncio.run(_ask_parties_async(question))


__all__ = ["Source", "PartyResponse", "WahlChatResponse", "ask_bundestag_parties"]
