from dataclasses import dataclass
from datetime import datetime


@dataclass
class PipelineContext:
    channel_id: int
    user_id: int | None
    timestamp: datetime
