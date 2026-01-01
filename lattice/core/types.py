from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class GhostTriggerReason(Enum):
    SCHEDULED = "scheduled"
    MILESTONE = "milestone"
    ENGAGEMENT_LAPSE = "engagement_lapse"
    CONTEXT_SWITCH = "context_switch"
    MANUAL = "manual"


class PipelineSourceType(Enum):
    USER = "user"
    GHOST = "ghost"
    SYSTEM = "system"


@dataclass
class GhostContext:
    user_id: int
    channel_id: int
    trigger_reason: GhostTriggerReason
    engagement_score: float
    last_ghost_at: datetime | None
    recent_message_count: int
    content_hash: str | None
    metadata: dict[str, Any]


@dataclass
class PipelineContext:
    source_type: PipelineSourceType
    channel_id: int
    user_id: int | None
    ghost_context: GhostContext | None
    timestamp: datetime
