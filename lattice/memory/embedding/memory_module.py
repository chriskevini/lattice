"""Embedding memory module - high-level interface for embedding-based memory."""

import structlog
from typing import Any
from uuid import UUID

from lattice.memory.embedding import PostgresEmbeddingMemoryRepository
from lattice.memory.procedural import get_prompt
from lattice.utils.json_parser import JSONParseError, parse_llm_json_response

logger = structlog.get_logger(__name__)

BATCH_SIZE = 18


class EmbeddingMemoryModule:
    """High-level interface for embedding-based memory operations."""

    def __init__(
        self,
        db_pool: Any,
        prompt_repo: Any | None = None,
    ) -> None:
        """Initialize the embedding memory module.

        Args:
            db_pool: Database connection pool
            prompt_repo: Optional prompt registry repository
        """
        self.repo = PostgresEmbeddingMemoryRepository(db_pool)
        self.prompt_repo = prompt_repo
        self.name = "embedding"

    async def retrieve_context(
        self,
        query: str,
        limit: int = 10,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant context for a query.

        Args:
            query: The query text
            limit: Maximum number of results
            threshold: Minimum similarity threshold

        Returns:
            Dict with keys: text (combined context), memories (list), count
        """
        memories = await self.repo.search_by_text(
            query_text=query,
            limit=limit,
            threshold=threshold,
        )

        if not memories:
            return {
                "text": "No relevant context found.",
                "memories": [],
                "count": 0,
            }

        lines = []
        for m in memories:
            content = m.get("content", "")
            similarity = m.get("similarity", 0)
            lines.append(f"- [{similarity:.2f}] {content}")

        text = "Relevant memories from embedding store:\n" + "\n".join(lines)

        return {
            "text": text,
            "memories": memories,
            "count": len(memories),
        }

    async def consolidate(
        self,
        messages: list[str],
        source_batch_id: str | None = None,
    ) -> int:
        """Consolidate messages into embedding memory.

        Args:
            messages: List of message texts to consolidate
            source_batch_id: Optional batch identifier

        Returns:
            Number of memories stored
        """
        if not messages:
            return 0

        from lattice.utils.llm_client import _LLMClient

        llm_client = _LLMClient()

        prompt_template = None
        if self.prompt_repo:
            prompt_template = await get_prompt(
                repo=self.prompt_repo,
                prompt_key="EMBEDDING_CONSOLIDATION",
            )

        if not prompt_template:
            prompt_text = DEFAULT_CONSOLIDATION_PROMPT

            prompt = f"""{prompt_text}

## Messages
{chr(10).join(f"[{i + 1}] {msg}" for i, msg in enumerate(messages))}

## Output Format
Return ONLY valid JSON. No prose.
{{"memories": ["memory 1", "memory 2", ...]}}
"""
        else:
            prompt = f"""Extract important information from the following messages as short memory summaries.

## Messages
{chr(10).join(f"[{i + 1}] {msg}" for i, msg in enumerate(messages))}

Extract 3-10 key facts or insights. Each memory should be a concise statement (1-2 sentences).

Return as JSON: {{"memories": ["memory 1", "memory 2", ...]}}
"""

        try:
            result = await llm_client.complete(
                prompt=prompt,
                temperature=0.2,
            )

            try:
                parsed = parse_llm_json_response(
                    content=result.content,
                    audit_result=None,
                    prompt_key="EMBEDDING_CONSOLIDATION",
                )
            except JSONParseError:
                logger.warning(
                    "Failed to parse consolidation response, trying extraction"
                )
                return 0

            memories = parsed.get("memories", [])
            if not memories:
                return 0

            embeddings = await llm_client.embed_batch(memories)

            items = [
                {"content": content, "embedding": emb}
                for content, emb in zip(memories, embeddings)
            ]

            count = await self.repo.store_memories_batch(
                items=items,
                source_batch_id=source_batch_id,
            )

            logger.info(
                "Consolidated memories to embedding store",
                batch_id=source_batch_id,
                count=count,
            )

            return count

        except Exception as e:
            logger.error("Failed to consolidate embeddings", exc_info=e)
            return 0

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        source_batch_id: str | None = None,
    ) -> UUID | None:
        """Store a single memory.

        Args:
            content: The memory content
            metadata: Optional metadata
            source_batch_id: Optional batch identifier

        Returns:
            UUID of stored memory or None if failed
        """
        from lattice.utils.llm_client import _LLMClient

        try:
            llm_client = _LLMClient()
            embedding = await llm_client.embed(content)

            return await self.repo.store_memory(
                content=content,
                embedding=embedding,
                metadata=metadata,
                source_batch_id=source_batch_id,
            )
        except Exception as e:
            logger.error("Failed to store memory", error=str(e))
            return None

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent memories.

        Args:
            limit: Maximum number of results

        Returns:
            List of recent memories
        """
        return await self.repo.get_recent(limit=limit)


DEFAULT_CONSOLIDATION_PROMPT = """Extract important information from the following messages as short memory summaries.

## Guidelines
- Extract key facts, preferences, goals, and insights
- Each memory should be a concise statement (1-2 sentences)
- Focus on information that would be useful for future context
"""
