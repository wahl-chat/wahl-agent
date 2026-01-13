"""Helpers for storing and retrieving data from Firestore."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.firestore_v1 import Client as FirestoreClient


_conversations_collection_name = os.getenv(
    "FIRESTORE_CONVERSATIONS_COLLECTION", "wahl_agent_conversations"
)
_topics_collection_name = os.getenv("FIRESTORE_TOPICS_COLLECTION", "wahl_agent_topics")
_credential_env_vars = ("FIREBASE_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS")

firestore_client: Optional[FirestoreClient] = None


def _initialize_firebase_app() -> firebase_admin.App:
    """Initialise the Firebase app once per process."""
    if firebase_admin._apps:
        return firebase_admin.get_app()

    cred_path = next(
        (
            os.getenv(var_name)
            for var_name in _credential_env_vars
            if os.getenv(var_name)
        ),
        None,
    )

    if cred_path:
        # Strip quotes and handle shell-escaped spaces
        cred_path = cred_path.strip("\"'")
        # Replace shell-escaped spaces with regular spaces
        cred_path = cred_path.replace("\\ ", " ")
        credentials_file = Path(cred_path).expanduser()
        if not credentials_file.exists():
            raise RuntimeError(
                f"Firebase credentials file '{credentials_file}' does not exist."
            )
        creds = credentials.Certificate(str(credentials_file))
        return firebase_admin.initialize_app(creds)

    try:
        return firebase_admin.initialize_app()
    except (ValueError, DefaultCredentialsError) as exc:
        raise RuntimeError(
            "Could not initialise Firebase. Provide a service-account JSON path via "
            "FIREBASE_CREDENTIALS_PATH or set GOOGLE_APPLICATION_CREDENTIALS."
        ) from exc


def get_firestore_client():
    """Return a cached Firestore client."""
    global firestore_client
    if firestore_client is not None:
        return firestore_client

    _initialize_firebase_app()
    firestore_client = firestore.client()
    return firestore_client


def save_conversation_metadata(
    *,
    topic: str,
    user_profile: Dict[str, Any],
    conversation_id: Optional[str] = None,
    stage: str = "start",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist the initial chat metadata to Firestore."""
    client = get_firestore_client()
    collection_ref = client.collection(_conversations_collection_name)
    doc_ref = (
        collection_ref.document(conversation_id)
        if conversation_id
        else collection_ref.document()
    )

    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "conversation_id": doc_ref.id,
        "topic": topic,
        "user_profile": user_profile,
        "stage": stage,
        "created_at": now,
        "updated_at": now,
        "started_at": now,
    }
    if extra:
        payload.update(extra)

    doc_ref.set(payload)
    return doc_ref.id


def update_conversation(
    *,
    conversation_id: str,
    stage: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Update specific fields in an existing conversation document."""
    client = get_firestore_client()
    doc_ref = client.collection(_conversations_collection_name).document(
        conversation_id
    )

    update_data: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
    if stage is not None:
        update_data["stage"] = stage
    if extra:
        update_data.update(extra)

    doc_ref.update(update_data)


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a conversation document by ID."""
    client: FirestoreClient = get_firestore_client()
    doc_ref = client.collection(_conversations_collection_name).document(
        conversation_id
    )
    doc = doc_ref.get()

    if not doc.exists:
        return None

    return doc.to_dict()


def get_party_positions_by_topic_id(
    topic_id: str,
) -> list[tuple[str, Dict[str, Any]]] | None:
    client = get_firestore_client()
    collection_ref = (
        client.collection(_topics_collection_name)
        .document(topic_id)
        .collection("party_positions")
    )

    docs = list(collection_ref.stream())
    if not docs:
        return None

    # Return as list of tuples (id, document), sorted by document[positionLeftToRight]
    return sorted(
        [(doc.id, doc.to_dict()) for doc in docs],
        key=lambda x: x[1]["positionLeftToRight"],
    )


__all__ = [
    "get_firestore_client",
    "save_conversation_metadata",
    "update_conversation",
    "get_conversation",
    "get_party_positions_by_topic_id",
]
