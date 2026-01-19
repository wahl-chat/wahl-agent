from flask import Flask, Response, jsonify, request
import json
from dotenv import load_dotenv

from src.agent_orchestrator import chat
from src.services.firestore_service import (
    save_conversation_metadata,
    get_party_positions_by_topic_id,
    get_conversation,
)

load_dotenv()
app = Flask(__name__)


@app.route("/chat-start", methods=["POST"])
def start_chat():
    payload = request.get_json(force=True) or {}
    required_fields = ["topic"]
    missing_fields = [field for field in required_fields if payload.get(field) is None]

    if missing_fields:
        return (
            jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}),
            400,
        )

    topic = payload["topic"]

    # Handle optional Prolific study metadata
    extra = None
    prolific_metadata = payload.get("prolific_metadata")
    if prolific_metadata:
        extra = {
            "prolific_metadata": prolific_metadata,
            "is_prolific_study": True,
        }

    try:
        conversation_id = save_conversation_metadata(
            topic=topic,
            extra=extra,
        )
    except RuntimeError as exc:
        app.logger.exception("Failed to store chat-start payload: %s", exc)
        return jsonify({"error": "Failed to store conversation metadata"}), 500

    return jsonify({"conversation_id": conversation_id}), 201


@app.route("/chat-stream", methods=["POST"])
def chat_stream():
    payload = request.get_json(force=True) or {}
    required_fields = ["user_message", "conversation_id"]
    missing_fields = [field for field in required_fields if payload.get(field) is None]
    if missing_fields:
        return (
            jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}),
            400,
        )
    user_message = payload.get("user_message")
    conversation_id = payload.get("conversation_id")

    return Response(
        (
            json.dumps(event).encode("utf-8") + b"\n"
            for event in chat(conversation_id, user_message)
        ),
        mimetype="text/event-stream",
    )


@app.route("/conversation-stage/<conversation_id>", methods=["GET"])
def get_conversation_stage(conversation_id: str):
    conversation = get_conversation(conversation_id)

    if conversation is None:
        return jsonify({"error": "Conversation not found"}), 404

    return jsonify({"stage": conversation.get("stage")}), 200


@app.route("/conversation-messages/<conversation_id>", methods=["GET"])
def get_conversation_messages(conversation_id: str):
    conversation = get_conversation(conversation_id)

    if conversation is None:
        return jsonify({"error": "Conversation not found"}), 404

    # Merge messages from all stages in chronological order
    all_messages = []
    stage_keys = [
        "active_listening_messages",
        "party_positioning_messages",
        "perspective_taking_messages",
        "deliberation_messages",
    ]

    for key in stage_keys:
        for msg in conversation.get(key, []):
            all_messages.append(
                {
                    "role": msg.get("type", "human"),
                    "content": msg.get("content", ""),
                }
            )

    # Add party matching result with sources
    party_matching_result = conversation.get("party_matching_result", None)
    if party_matching_result:
        party_matching_sources = conversation.get("party_matching_sources", [])
        all_messages.append({
            "role": "assistant",
            "content": party_matching_result,
            "sources": party_matching_sources,
        })

    return jsonify({"messages": all_messages}), 200


@app.route("/conversation-topic/<conversation_id>", methods=["GET"])
def get_conversation_topic(conversation_id: str):
    conversation = get_conversation(conversation_id)
    
    if conversation is None:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"topic": conversation.get("topic")}), 200

