import asyncio
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class RecursionGuard:
    MAX_GHOST_DEPTH: int = 2

    def __init__(self) -> None:
        self._depth: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def enter_ghost(self) -> bool:
        async with self._lock:
            if self._depth >= self.MAX_GHOST_DEPTH:
                logger.warning("Max ghost recursion depth exceeded")
                return False
            self._depth += 1
            return True

    def exit_ghost(self) -> None:
        self._depth = max(0, self._depth - 1)
