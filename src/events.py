from enum import Enum


class EventType(Enum):
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    MESSAGE_CHUNK = "message_chunk"
    PROGRESS_UPDATE = "progress_update"
    SOURCES_READY = "sources_ready"
    END = "end"
