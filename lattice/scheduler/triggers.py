import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class TriggerType(Enum):
    MILESTONE = "milestone"
    ENGAGEMENT_LAPSE = "engagement_lapse"
    CONTEXT_SWITCH = "context_switch"
    MANUAL = "manual"


@dataclass
class TriggerEvent:
    trigger_type: TriggerType
    user_id: int
    channel_id: int
    timestamp: datetime
    metadata: dict[str, Any]


class TriggerDetector:
    ENGAGEMENT_LAPSE_HOURS: int = 72
    MILESTONE_KEYWORDS: list[str] = [
        "completed",
        "achieved",
        "finished",
        "done",
        "reached",
        "goal",
        "milestone",
    ]

    def __init__(self, db_pool: Any) -> None:
        self.db_pool = db_pool

    async def check_engagement_lapse(self, user_id: int) -> bool:
        async with self.db_pool.acquire() as conn:
            last_activity = await conn.fetchval(
                """
                SELECT MAX(timestamp) FROM raw_messages
                WHERE user_id = $1 AND is_bot = false
                """,
                user_id,
            )

        if not last_activity:
            return False

        lapse_threshold = datetime.utcnow() - timedelta(hours=self.ENGAGEMENT_LAPSE_HOURS)
        return last_activity < lapse_threshold

    async def check_milestone(self, content: str) -> bool:
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self.MILESTONE_KEYWORDS)

    async def detect_triggers(
        self,
        user_id: int,
        channel_id: int,
        recent_messages: list[dict[str, Any]],
    ) -> list[TriggerEvent]:
        triggers: list[TriggerEvent] = []

        if await self.check_engagement_lapse(user_id):
            triggers.append(
                TriggerEvent(
                    trigger_type=TriggerType.ENGAGEMENT_LAPSE,
                    user_id=user_id,
                    channel_id=channel_id,
                    timestamp=datetime.utcnow(),
                    metadata={"reason": "No activity for 72+ hours"},
                )
            )

        for msg in recent_messages:
            if await self.check_milestone(msg.get("content", "")):
                triggers.append(
                    TriggerEvent(
                        trigger_type=TriggerType.MILESTONE,
                        user_id=user_id,
                        channel_id=channel_id,
                        timestamp=datetime.utcnow(),
                        metadata={"message_id": msg.get("id")},
                    )
                )
                break

        return triggers

    async def should_trigger_proactive(
        self,
        user_id: int,
        channel_id: int,
        last_ghost_at: datetime | None,
    ) -> tuple[bool, TriggerEvent | None]:
        if last_ghost_at is None:
            return True, None

        async with self.db_pool.acquire() as conn:
            next_scheduled = await conn.fetchval(
                "SELECT next_scheduled_at FROM proactive_schedule_config WHERE user_id = $1",
                user_id,
            )

        if not next_scheduled:
            return True, None

        return datetime.utcnow() >= next_scheduled, None
