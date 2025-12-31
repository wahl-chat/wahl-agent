from enum import Enum


class EventType(Enum):
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    MESSAGE_CHUNK = "message_chunk"
    END = "end"
