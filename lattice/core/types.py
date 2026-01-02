from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class PipelineSourceType(Enum):
    USER = "user"
    PROACTIVE = "proactive"
    SYSTEM = "system"


@dataclass
class PipelineContext:
    source_type: PipelineSourceType
    channel_id: int
    user_id: int | None
    timestamp: datetime
