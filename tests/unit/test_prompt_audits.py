"""Unit tests for prompt_audits module.

Tests the prompt audit storage, retrieval, and feedback linking
functionality for the Dreaming Cycle.
"""

from datetime import UTC, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from lattice.memory.prompt_audits import (
    PromptAudit,
    get_audit_by_dream_message,
    get_audits_with_feedback,
    link_feedback_to_audit,
    link_feedback_to_audit_by_id,
    store_prompt_audit,
    update_audit_dream_message,
)
from lattice.memory.repositories import PromptAuditRepository


class TestPromptAuditDataclass:
    """Tests for PromptAudit dataclass."""

    def test_prompt_audit_creation(self) -> None:
        """Test creating a PromptAudit instance."""
        audit = PromptAudit(
            audit_id=UUID("12345678-1234-1234-1234-123456789abc"),
            prompt_key="test_prompt",
            template_version=1,
            message_id=None,
            rendered_prompt="Hello {name}",
            response_content="Hello World",
            model="gpt-4",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=Decimal("0.001"),
            latency_ms=100,
            finish_reason="stop",
            cache_discount_usd=None,
            native_tokens_cached=None,
            native_tokens_reasoning=None,
            upstream_id=None,
            cancelled=None,
            moderation_latency_ms=None,
            execution_metadata=None,
            archetype_matched=None,
            archetype_confidence=None,
            reasoning=None,
            main_discord_message_id=123,
            dream_discord_message_id=456,
            feedback_id=None,
            created_at=datetime.now(timezone.utc),
        )

        assert audit.prompt_key == "test_prompt"
        assert audit.template_version == 1
        assert audit.model == "gpt-4"
        assert audit.prompt_tokens == 10

    def test_prompt_audit_with_none_values(self) -> None:
        """Test creating PromptAudit with None optional values."""
        audit = PromptAudit(
            audit_id=UUID("12345678-1234-1234-1234-123456789abc"),
            prompt_key="test_prompt",
            template_version=None,
            message_id=None,
            rendered_prompt=None,
            response_content="Response",
            model=None,
            provider=None,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
            latency_ms=None,
            finish_reason=None,
            cache_discount_usd=None,
            native_tokens_cached=None,
            native_tokens_reasoning=None,
            upstream_id=None,
            cancelled=None,
            moderation_latency_ms=None,
            execution_metadata=None,
            archetype_matched=None,
            archetype_confidence=None,
            reasoning=None,
            main_discord_message_id=None,
            dream_discord_message_id=None,
            feedback_id=None,
            created_at=datetime.now(timezone.utc),
        )

        assert audit.template_version is None
        assert audit.model is None

    def test_prompt_audit_equality(self) -> None:
        """Test PromptAudit equality comparison."""
        created_at = datetime(2026, 1, 12, 10, 30, 0, tzinfo=timezone.utc)
        audit1 = PromptAudit(
            audit_id=UUID("12345678-1234-1234-1234-123456789abc"),
            prompt_key="test",
            template_version=1,
            message_id=None,
            rendered_prompt="prompt",
            response_content="response",
            model=None,
            provider=None,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
            latency_ms=None,
            finish_reason=None,
            cache_discount_usd=None,
            native_tokens_cached=None,
            native_tokens_reasoning=None,
            upstream_id=None,
            cancelled=None,
            moderation_latency_ms=None,
            execution_metadata=None,
            archetype_matched=None,
            archetype_confidence=None,
            reasoning=None,
            main_discord_message_id=None,
            dream_discord_message_id=None,
            feedback_id=None,
            created_at=created_at,
        )

        audit2 = PromptAudit(
            audit_id=UUID("12345678-1234-1234-1234-123456789abc"),
            prompt_key="test",
            template_version=1,
            message_id=None,
            rendered_prompt="prompt",
            response_content="response",
            model=None,
            provider=None,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
            latency_ms=None,
            finish_reason=None,
            cache_discount_usd=None,
            native_tokens_cached=None,
            native_tokens_reasoning=None,
            upstream_id=None,
            cancelled=None,
            moderation_latency_ms=None,
            execution_metadata=None,
            archetype_matched=None,
            archetype_confidence=None,
            reasoning=None,
            main_discord_message_id=None,
            dream_discord_message_id=None,
            feedback_id=None,
            created_at=created_at,
        )

        assert audit1 == audit2

    def test_prompt_audit_inequality(self) -> None:
        """Test PromptAudit inequality when fields differ."""
        created_at = datetime(2026, 1, 12, 10, 30, 0, tzinfo=timezone.utc)
        audit1 = PromptAudit(
            audit_id=UUID("12345678-1234-1234-1234-123456789abc"),
            prompt_key="test1",
            template_version=1,
            message_id=None,
            rendered_prompt="prompt",
            response_content="response",
            model=None,
            provider=None,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
            latency_ms=None,
            finish_reason=None,
            cache_discount_usd=None,
            native_tokens_cached=None,
            native_tokens_reasoning=None,
            upstream_id=None,
            cancelled=None,
            moderation_latency_ms=None,
            execution_metadata=None,
            archetype_matched=None,
            archetype_confidence=None,
            reasoning=None,
            main_discord_message_id=None,
            dream_discord_message_id=None,
            feedback_id=None,
            created_at=created_at,
        )

        audit2 = PromptAudit(
            audit_id=UUID("12345678-1234-1234-1234-123456789abc"),
            prompt_key="test2",
            template_version=1,
            message_id=None,
            rendered_prompt="prompt",
            response_content="response",
            model=None,
            provider=None,
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
            latency_ms=None,
            finish_reason=None,
            cache_discount_usd=None,
            native_tokens_cached=None,
            native_tokens_reasoning=None,
            upstream_id=None,
            cancelled=None,
            moderation_latency_ms=None,
            execution_metadata=None,
            archetype_matched=None,
            archetype_confidence=None,
            reasoning=None,
            main_discord_message_id=None,
            dream_discord_message_id=None,
            feedback_id=None,
            created_at=created_at,
        )

        assert audit1 != audit2

    def test_prompt_audit_all_openrouter_fields(self) -> None:
        """Test PromptAudit with all OpenRouter telemetry fields."""
        audit = PromptAudit(
            audit_id=UUID("12345678-1234-1234-1234-123456789abc"),
            prompt_key="TEST",
            template_version=1,
            message_id=None,
            rendered_prompt="prompt",
            response_content="response",
            model="openrouter/model",
            provider="openrouter",
            prompt_tokens=100,
            completion_tokens=200,
            cost_usd=Decimal("0.005"),
            latency_ms=500,
            finish_reason="stop",
            cache_discount_usd=Decimal("0.001"),
            native_tokens_cached=50,
            native_tokens_reasoning=100,
            upstream_id="upstream-123",
            cancelled=False,
            moderation_latency_ms=25,
            execution_metadata=None,
            archetype_matched="reasoning",
            archetype_confidence=0.95,
            reasoning={"steps": ["analyze", "plan"]},
            main_discord_message_id=123,
            dream_discord_message_id=456,
            feedback_id=None,
            created_at=datetime.now(timezone.utc),
        )

        assert audit.cache_discount_usd == Decimal("0.001")
        assert audit.native_tokens_cached == 50
        assert audit.native_tokens_reasoning == 100
        assert audit.upstream_id == "upstream-123"
        assert audit.cancelled is False
        assert audit.moderation_latency_ms == 25


class TestStorePromptAudit:
    """Tests for store_prompt_audit function."""

    @pytest.mark.asyncio
    async def test_store_prompt_audit_success(self) -> None:
        """Test successfully storing a prompt audit."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        audit_id = UUID("12345678-1234-5678-1234-567812345678")
        mock_repo.store_audit = AsyncMock(return_value=audit_id)

        result = await store_prompt_audit(
            repo=mock_repo,
            prompt_key="BASIC_RESPONSE",
            rendered_prompt="Test prompt",
            response_content="Test response",
            main_discord_message_id=123456789,
            template_version=1,
            message_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            model="gpt-4o-mini",
            provider="openrouter",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.001,
            latency_ms=250,
            execution_metadata={
                "episodic": 5,
                "semantic": 3,
                "graph": 0,
            },
        )

        assert result == audit_id
        mock_repo.store_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_prompt_audit_no_message_id(self) -> None:
        """Test storing prompt audit with None for main_discord_message_id."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        audit_id = UUID("aaaa0000-0000-0000-0000-000000000000")
        mock_repo.store_audit = AsyncMock(return_value=audit_id)

        result = await store_prompt_audit(
            repo=mock_repo,
            prompt_key="CONTEXTUAL_NUDGE",
            rendered_prompt="Nudge prompt",
            response_content="Nudge response",
            main_discord_message_id=None,
        )

        assert result == audit_id
        mock_repo.store_audit.assert_called_once()


class TestUpdateAuditDreamMessage:
    """Tests for update_audit_dream_message function."""

    @pytest.mark.asyncio
    async def test_update_audit_dream_message_success(self) -> None:
        """Test successfully updating dream message ID."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.update_dream_message = AsyncMock(return_value=True)

        result = await update_audit_dream_message(
            repo=mock_repo,
            audit_id=UUID("12345678-1234-5678-1234-567812345678"),
            dream_discord_message_id=555555555,
        )

        assert result is True
        mock_repo.update_dream_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_audit_dream_message_not_found(self) -> None:
        """Test updating non-existent audit returns False."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.update_dream_message = AsyncMock(return_value=False)

        result = await update_audit_dream_message(
            repo=mock_repo,
            audit_id=UUID("99999999-9999-9999-9999-999999999999"),
            dream_discord_message_id=111111111,
        )

        assert result is False


class TestLinkFeedbackToAudit:
    """Tests for link_feedback_to_audit function."""

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_success(self) -> None:
        """Test successfully linking feedback to audit."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.link_feedback = AsyncMock(return_value=True)

        result = await link_feedback_to_audit(
            repo=mock_repo,
            dream_discord_message_id=777777777,
            feedback_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        )

        assert result is True
        mock_repo.link_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_not_found(self) -> None:
        """Test linking feedback when audit not found."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.link_feedback = AsyncMock(return_value=False)

        result = await link_feedback_to_audit(
            repo=mock_repo,
            dream_discord_message_id=999999999,
            feedback_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        )

        assert result is False


class TestLinkFeedbackToAuditById:
    """Tests for link_feedback_to_audit_by_id function."""

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_by_id_success(self) -> None:
        """Test successfully linking feedback to audit by audit_id."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.link_feedback_by_id = AsyncMock(return_value=True)

        result = await link_feedback_to_audit_by_id(
            repo=mock_repo,
            audit_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            feedback_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        )

        assert result is True
        mock_repo.link_feedback_by_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_by_id_not_found(self) -> None:
        """Test linking feedback when audit not found."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.link_feedback_by_id = AsyncMock(return_value=False)

        result = await link_feedback_to_audit_by_id(
            repo=mock_repo,
            audit_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            feedback_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        )

        assert result is False


class TestGetAuditByDreamMessage:
    """Tests for get_audit_by_dream_message function."""

    @pytest.mark.asyncio
    async def test_get_audit_by_dream_message_found(self) -> None:
        """Test retrieving audit by dream message ID."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        audit_data = {
            "id": UUID("12345678-1234-5678-1234-567812345678"),
            "prompt_key": "BASIC_RESPONSE",
            "template_version": 1,
            "message_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            "rendered_prompt": "Test prompt",
            "response_content": "Test response",
            "model": "gpt-4o-mini",
            "provider": "openrouter",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_usd": 0.001,
            "latency_ms": 250,
            "finish_reason": "stop",
            "cache_discount_usd": 0.0001,
            "native_tokens_cached": 10,
            "native_tokens_reasoning": 5,
            "upstream_id": "upstream-123",
            "cancelled": False,
            "moderation_latency_ms": 50,
            "execution_metadata": {"episodic": 5, "semantic": 3, "graph": 0},
            "archetype_matched": None,
            "archetype_confidence": None,
            "reasoning": None,
            "main_discord_message_id": 123456789,
            "dream_discord_message_id": 987654321,
            "feedback_id": UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            "created_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        }
        mock_repo.get_by_dream_message = AsyncMock(return_value=audit_data)

        result = await get_audit_by_dream_message(
            repo=mock_repo, dream_discord_message_id=987654321
        )

        assert result is not None
        assert result.audit_id == UUID("12345678-1234-5678-1234-567812345678")
        assert result.rendered_prompt == "Test prompt"
        assert result.dream_discord_message_id == 987654321
        mock_repo.get_by_dream_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_audit_by_dream_message_not_found(self) -> None:
        """Test retrieving non-existent audit returns None."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.get_by_dream_message = AsyncMock(return_value=None)

        result = await get_audit_by_dream_message(
            repo=mock_repo, dream_discord_message_id=999999999
        )

        assert result is None


class TestGetAuditsWithFeedback:
    """Tests for get_audits_with_feedback function."""

    @pytest.mark.asyncio
    async def test_get_audits_with_feedback_found(self) -> None:
        """Test retrieving audits with feedback."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        audit_data_list = [
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "prompt_key": "BASIC_RESPONSE",
                "template_version": 1,
                "message_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                "rendered_prompt": "Prompt 1",
                "response_content": "Response 1",
                "model": "gpt-4o-mini",
                "provider": "openrouter",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost_usd": 0.001,
                "latency_ms": 250,
                "finish_reason": "stop",
                "cache_discount_usd": 0.0001,
                "native_tokens_cached": 10,
                "native_tokens_reasoning": None,
                "upstream_id": "upstream-111",
                "cancelled": False,
                "moderation_latency_ms": 20,
                "execution_metadata": {"episodic": 5, "semantic": 3, "graph": 0},
                "archetype_matched": None,
                "archetype_confidence": None,
                "reasoning": None,
                "main_discord_message_id": 111,
                "dream_discord_message_id": 222,
                "feedback_id": UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
                "created_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            },
            {
                "id": UUID("22222222-2222-2222-2222-222222222222"),
                "prompt_key": "BASIC_RESPONSE",
                "template_version": 1,
                "message_id": UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                "rendered_prompt": "Prompt 2",
                "response_content": "Response 2",
                "model": "gpt-4o-mini",
                "provider": "openrouter",
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "cost_usd": 0.002,
                "latency_ms": 300,
                "finish_reason": "stop",
                "cache_discount_usd": None,
                "native_tokens_cached": None,
                "native_tokens_reasoning": None,
                "upstream_id": "upstream-222",
                "cancelled": False,
                "moderation_latency_ms": None,
                "execution_metadata": {"episodic": 3, "semantic": 2, "graph": 1},
                "archetype_matched": None,
                "archetype_confidence": None,
                "reasoning": None,
                "main_discord_message_id": 333,
                "dream_discord_message_id": 444,
                "feedback_id": UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
                "created_at": datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            },
        ]
        mock_repo.get_with_feedback = AsyncMock(return_value=audit_data_list)

        result = await get_audits_with_feedback(repo=mock_repo, limit=10)

        assert len(result) == 2
        assert result[0].audit_id == UUID("11111111-1111-1111-1111-111111111111")
        assert result[1].audit_id == UUID("22222222-2222-2222-2222-222222222222")
        mock_repo.get_with_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_audits_with_feedback_empty(self) -> None:
        """Test retrieving audits when none have feedback."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.get_with_feedback = AsyncMock(return_value=[])

        result = await get_audits_with_feedback(repo=mock_repo, limit=10)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_audits_with_feedback_custom_limit(self) -> None:
        """Test retrieving audits with custom limit."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.get_with_feedback = AsyncMock(return_value=[])

        result = await get_audits_with_feedback(repo=mock_repo, limit=5)

        assert len(result) == 0
        mock_repo.get_with_feedback.assert_called_once_with(5, 0)


class TestPromptAuditEdgeCases:
    """Edge case tests for prompt audit functions."""

    @pytest.mark.asyncio
    async def test_store_audit_with_empty_response(self) -> None:
        """Test storing audit with empty response content."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.store_audit = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        result = await store_prompt_audit(
            repo=mock_repo,
            prompt_key="TEST",
            response_content="",
            main_discord_message_id=123,
        )

        assert result == UUID("12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_store_audit_with_special_characters(self) -> None:
        """Test storing audit with special characters in content."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.store_audit = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        special_content = "Hello! 🌍\n\n```python\nprint('world')\n```"
        result = await store_prompt_audit(
            repo=mock_repo,
            prompt_key="TEST",
            response_content=special_content,
            main_discord_message_id=123,
        )

        assert result == UUID("12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_store_audit_with_json_metadata(self) -> None:
        """Test storing audit with JSON execution metadata."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.store_audit = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        metadata = {
            "context_config": {"channels": [1, 2, 3]},
            "entities_found": ["User", "Project"],
        }
        result = await store_prompt_audit(
            repo=mock_repo,
            prompt_key="TEST",
            response_content="response",
            main_discord_message_id=123,
            execution_metadata=metadata,
        )

        assert result == UUID("12345678-1234-1234-1234-123456789abc")

    @pytest.mark.asyncio
    async def test_get_audit_by_dream_message_with_complex_row(self) -> None:
        """Test retrieving audit with complex data types."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_row = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "prompt_key": "COMPLEX_PROMPT",
            "template_version": 5,
            "message_id": UUID("22222222-2222-2222-2222-222222222222"),
            "rendered_prompt": '{"name": "test"}',
            "response_content": '{"result": "success"}',
            "model": "claude-3-opus",
            "provider": "anthropic",
            "prompt_tokens": 500,
            "completion_tokens": 1000,
            "cost_usd": 0.05,
            "latency_ms": 2000,
            "finish_reason": "stop",
            "cache_discount_usd": 0.01,
            "native_tokens_cached": 200,
            "native_tokens_reasoning": 500,
            "upstream_id": "up-12345",
            "cancelled": False,
            "moderation_latency_ms": 25,
            "execution_metadata": {"cache_hit": True},
            "archetype_matched": "reasoning",
            "archetype_confidence": 0.88,
            "reasoning": {"steps": ["analyze", "plan", "execute"]},
            "main_discord_message_id": 100,
            "dream_discord_message_id": 200,
            "feedback_id": None,
            "created_at": datetime(2026, 1, 12, 10, 30, 0, tzinfo=timezone.utc),
        }
        mock_repo.get_by_dream_message = AsyncMock(return_value=mock_row)

        result = await get_audit_by_dream_message(
            repo=mock_repo,
            dream_discord_message_id=200,
        )

        assert result is not None
        assert result.prompt_key == "COMPLEX_PROMPT"
        assert result.template_version == 5
        assert result.execution_metadata == {"cache_hit": True}
        assert result.reasoning == {"steps": ["analyze", "plan", "execute"]}

    @pytest.mark.asyncio
    async def test_get_audits_with_feedback_multiple_results(self) -> None:
        """Test retrieving multiple audits with feedback."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_rows = [
            {
                "id": UUID("11111111-1111-1111-1111-111111111111"),
                "prompt_key": "PROMPT_A",
                "template_version": 1,
                "message_id": None,
                "rendered_prompt": "prompt A",
                "response_content": "response A",
                "model": None,
                "provider": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cost_usd": None,
                "latency_ms": None,
                "finish_reason": None,
                "cache_discount_usd": None,
                "native_tokens_cached": None,
                "native_tokens_reasoning": None,
                "upstream_id": None,
                "cancelled": None,
                "moderation_latency_ms": None,
                "execution_metadata": None,
                "archetype_matched": None,
                "archetype_confidence": None,
                "reasoning": None,
                "main_discord_message_id": None,
                "dream_discord_message_id": None,
                "feedback_id": UUID("33333333-3333-3333-3333-333333333333"),
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": UUID("22222222-2222-2222-2222-222222222222"),
                "prompt_key": "PROMPT_B",
                "template_version": 2,
                "message_id": None,
                "rendered_prompt": "prompt B",
                "response_content": "response B",
                "model": None,
                "provider": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cost_usd": None,
                "latency_ms": None,
                "finish_reason": None,
                "cache_discount_usd": None,
                "native_tokens_cached": None,
                "native_tokens_reasoning": None,
                "upstream_id": None,
                "cancelled": None,
                "moderation_latency_ms": None,
                "execution_metadata": None,
                "archetype_matched": None,
                "archetype_confidence": None,
                "reasoning": None,
                "main_discord_message_id": None,
                "dream_discord_message_id": None,
                "feedback_id": UUID("44444444-4444-4444-4444-444444444444"),
                "created_at": datetime.now(timezone.utc),
            },
        ]
        mock_repo.get_with_feedback = AsyncMock(return_value=mock_rows)

        result = await get_audits_with_feedback(repo=mock_repo, limit=100)

        assert len(result) == 2
        assert result[0].prompt_key == "PROMPT_A"
        assert result[1].prompt_key == "PROMPT_B"

    @pytest.mark.asyncio
    async def test_store_audit_various_finish_reasons(self) -> None:
        """Test storing audit with different finish reasons."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.store_audit = AsyncMock(
            return_value=UUID("12345678-1234-1234-1234-123456789abc")
        )

        for reason in ["stop", "length", "content_filter", "tool_calls"]:
            await store_prompt_audit(
                repo=mock_repo,
                prompt_key="TEST",
                response_content="response",
                main_discord_message_id=123,
                finish_reason=reason,
            )

        assert mock_repo.store_audit.call_count == 4

    @pytest.mark.asyncio
    async def test_update_audit_dream_message_multiple_calls(self) -> None:
        """Test updating dream message ID multiple times."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.update_dream_message = AsyncMock(return_value=True)

        audit_id = UUID("12345678-1234-1234-1234-123456789abc")

        result1 = await update_audit_dream_message(
            repo=mock_repo,
            audit_id=audit_id,
            dream_discord_message_id=100,
        )

        result2 = await update_audit_dream_message(
            repo=mock_repo,
            audit_id=audit_id,
            dream_discord_message_id=200,
        )

        assert result1 is True
        assert result2 is True
        assert mock_repo.update_dream_message.call_count == 2

    @pytest.mark.asyncio
    async def test_link_feedback_different_message_ids(self) -> None:
        """Test linking feedback for different dream message IDs."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_repo.link_feedback = AsyncMock(return_value=True)

        for i in range(5):
            feedback_uuid = UUID(f"{i * 4 + 1:08d}0000-0000-0000-000000000000")
            result = await link_feedback_to_audit(
                repo=mock_repo,
                dream_discord_message_id=100 + i,
                feedback_id=feedback_uuid,
            )
            assert result is True

        assert mock_repo.link_feedback.call_count == 5

    @pytest.mark.asyncio
    async def test_get_audit_by_dream_message_with_none_fields(self) -> None:
        """Test retrieving audit where many optional fields are None."""
        mock_repo = MagicMock(spec=PromptAuditRepository)
        mock_row = {
            "id": UUID("12345678-1234-1234-1234-123456789abc"),
            "prompt_key": "MINIMAL_PROMPT",
            "template_version": None,
            "message_id": None,
            "rendered_prompt": None,
            "response_content": "Simple response",
            "model": None,
            "provider": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
            "latency_ms": None,
            "finish_reason": None,
            "cache_discount_usd": None,
            "native_tokens_cached": None,
            "native_tokens_reasoning": None,
            "upstream_id": None,
            "cancelled": None,
            "moderation_latency_ms": None,
            "execution_metadata": None,
            "archetype_matched": None,
            "archetype_confidence": None,
            "reasoning": None,
            "main_discord_message_id": None,
            "dream_discord_message_id": 123,
            "feedback_id": None,
            "created_at": datetime.now(timezone.utc),
        }
        mock_repo.get_by_dream_message = AsyncMock(return_value=mock_row)

        result = await get_audit_by_dream_message(
            repo=mock_repo,
            dream_discord_message_id=123,
        )

        assert result is not None
        assert result.prompt_key == "MINIMAL_PROMPT"
        assert result.template_version is None
        assert result.rendered_prompt is None
        assert result.model is None
