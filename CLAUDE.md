# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

wahl-agent-backend is a Flask-based conversational AI application for German political education. It uses LangChain/LangGraph to guide voters through a multi-stage dialogue about political party positions, helping users understand different perspectives and find party alignment.

## Development Commands

```bash
# Install dependencies
poetry install

# Run locally with hot reloading
poetry run flask --app src/controller.py run --debug

# Format code (ruff, 88 char line length)
poetry run ruff format .
```

## Architecture

### Conversation State Machine

The application follows a 7-stage conversation flow defined in `src/conversation/conversation_state.py`:

```
START → ACTIVE_LISTENING → PARTY_POSITIONING → PERSPECTIVE_TAKING → DELIBERATION → PARTY_MATCHING → END
```

Each stage has its own module in `src/stages/` containing:
- LLM agent with stage-specific system prompt
- Tool definitions that trigger stage transitions
- State management for messages and summaries

### Stage Transitions via Tools

Stages advance through LangChain `@tool` decorated functions that:
1. Store the stage summary in Firestore
2. Update the conversation stage
3. Allow the LLM to control when phases complete

Example tools: `end_active_listening()`, `end_party_positioning()`, etc.

### Routing

`src/agent_orchestrator.py` uses Python match statements to route requests to the appropriate stage handler based on `ConversationState.stage`.

### Real-time Streaming

Responses stream via Server-Sent Events (SSE) using Flask's generator-based `Response()`. Event types defined in `src/events.py`: `MESSAGE_START`, `MESSAGE_CHUNK`, `MESSAGE_END`, `END`.

### External Integrations

- **Firestore**: Stores conversation state in `wahl_agent_conversations` collection, party positions in `wahl_agent_topics`
- **wahl-chat-backend**: WebSocket client (`src/services/wahl_chat_service.py`) fetches party responses for matching
- **Perplexity API**: Used as a tool in perspective-taking stage for fact retrieval

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat-start` | POST | Create conversation (topic, user profile) |
| `/chat-stream` | POST | Stream conversation messages (SSE) |
| `/conversation-stage/<id>` | GET | Get current stage |
| `/conversation-messages/<id>` | GET | Get all messages |
| `/conversation-topic/<id>` | GET | Get topic |

## Key Files

- `src/controller.py` - Flask app and endpoints
- `src/agent_orchestrator.py` - Stage routing logic
- `src/prompts.py` - All system prompts for each stage
- `src/conversation/conversation_state.py` - ConversationState and ConversationStage classes
- `src/services/firestore_service.py` - Firestore CRUD operations
- `src/utils/messages.py` - LangChain message serialization

## Environment Variables

Required in `.env` (see `.env.example`):
- `OPENAI_MODEL` - Model identifier (supports custom base URL)
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` - Optional custom LLM router
- `FIREBASE_CREDENTIALS_PATH` - Path to service account JSON
- `FIRESTORE_CONVERSATIONS_COLLECTION`
- `FIRESTORE_TOPICS_COLLECTION`
- `WAHL_CHAT_BACKEND_URL` - WebSocket endpoint for party matching

## Code Style

- Ruff for formatting and linting
- 88 character line length
- Double quotes for strings
- Python 3.11-3.12

## Git Use
- Never commit or push anything unless stated or approved by the user.
- Never add claude code as co-author of the commits (unless stated otherwise).
- Don't say that the commit was generated using claude code.

