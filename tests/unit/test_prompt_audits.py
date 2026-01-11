"""Unit tests for prompt_audits module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from lattice.memory.prompt_audits import (
    get_audit_by_dream_message,
    get_audits_with_feedback,
    link_feedback_to_audit,
    link_feedback_to_audit_by_id,
    store_prompt_audit,
    update_audit_dream_message,
)


def _create_mock_pool(mock_conn: MagicMock) -> MagicMock:
    """Create a mock database pool with the given connection."""
    mock_pool = MagicMock()
    mock_pool.pool = mock_pool
    mock_acquire_cm = MagicMock()
    mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire_cm.__aexit__ = AsyncMock()
    mock_pool.pool.acquire = MagicMock(return_value=mock_acquire_cm)
    return mock_pool


class TestStorePromptAudit:
    """Tests for store_prompt_audit function."""

    @pytest.mark.asyncio
    async def test_store_prompt_audit_success(self) -> None:
        """Test successfully storing a prompt audit."""
        mock_conn = MagicMock()
        audit_id = UUID("12345678-1234-5678-1234-567812345678")

        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda key: {"id": audit_id}[key]
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = _create_mock_pool(mock_conn)

        result = await store_prompt_audit(
            db_pool=mock_pool,
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
            context_config={
                "episodic": 5,
                "semantic": 3,
                "graph": 0,
            },
        )

        assert result == audit_id
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_prompt_audit_no_message_id(self) -> None:
        """Test storing prompt audit with None for main_discord_message_id."""
        mock_conn = MagicMock()
        audit_id = UUID("aaaa0000-0000-0000-0000-000000000000")

        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda key: {"id": audit_id}[key]
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = _create_mock_pool(mock_conn)

        result = await store_prompt_audit(
            db_pool=mock_pool,
            prompt_key="CONTEXTUAL_NUDGE",
            rendered_prompt="Nudge prompt",
            response_content="Nudge response",
            main_discord_message_id=None,
        )

        assert result == audit_id
        mock_conn.fetchrow.assert_called_once()


class TestUpdateAuditDreamMessage:
    """Tests for update_audit_dream_message function."""

    @pytest.mark.asyncio
    async def test_update_audit_dream_message_success(self) -> None:
        """Test successfully updating dream message ID."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        mock_pool = _create_mock_pool(mock_conn)

        result = await update_audit_dream_message(
            db_pool=mock_pool,
            audit_id=UUID("12345678-1234-5678-1234-567812345678"),
            dream_discord_message_id=555555555,
        )

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_audit_dream_message_not_found(self) -> None:
        """Test updating non-existent audit returns False."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")

        mock_pool = _create_mock_pool(mock_conn)

        result = await update_audit_dream_message(
            db_pool=mock_pool,
            audit_id=UUID("99999999-9999-9999-9999-999999999999"),
            dream_discord_message_id=111111111,
        )

        assert result is False


class TestLinkFeedbackToAudit:
    """Tests for link_feedback_to_audit function."""

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_success(self) -> None:
        """Test successfully linking feedback to audit."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        mock_pool = _create_mock_pool(mock_conn)

        result = await link_feedback_to_audit(
            db_pool=mock_pool,
            dream_discord_message_id=777777777,
            feedback_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        )

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_not_found(self) -> None:
        """Test linking feedback when audit not found."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")

        mock_pool = _create_mock_pool(mock_conn)

        result = await link_feedback_to_audit(
            db_pool=mock_pool,
            dream_discord_message_id=999999999,
            feedback_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        )

        assert result is False


class TestLinkFeedbackToAuditById:
    """Tests for link_feedback_to_audit_by_id function."""

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_by_id_success(self) -> None:
        """Test successfully linking feedback to audit by audit_id."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")

        mock_pool = _create_mock_pool(mock_conn)

        result = await link_feedback_to_audit_by_id(
            db_pool=mock_pool,
            audit_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            feedback_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        )

        assert result is True
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "UPDATE prompt_audits" in call_args[0][0]
        assert call_args[0][1] == UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
        assert call_args[0][2] == UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    @pytest.mark.asyncio
    async def test_link_feedback_to_audit_by_id_not_found(self) -> None:
        """Test linking feedback when audit not found."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")

        mock_pool = _create_mock_pool(mock_conn)

        result = await link_feedback_to_audit_by_id(
            db_pool=mock_pool,
            audit_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            feedback_id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        )

        assert result is False


class TestGetAuditByDreamMessage:
    """Tests for get_audit_by_dream_message function."""

    @pytest.mark.asyncio
    async def test_get_audit_by_dream_message_found(self) -> None:
        """Test retrieving audit by dream message ID."""
        mock_conn = MagicMock()
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
            "context_config": {"episodic": 5, "semantic": 3, "graph": 0},
            "archetype_matched": None,
            "archetype_confidence": None,
            "reasoning": None,
            "main_discord_message_id": 123456789,
            "dream_discord_message_id": 987654321,
            "feedback_id": UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            "created_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        }
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda key: audit_data[key]
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = _create_mock_pool(mock_conn)

        result = await get_audit_by_dream_message(
            db_pool=mock_pool, dream_discord_message_id=987654321
        )

        assert result is not None
        assert result.audit_id == UUID("12345678-1234-5678-1234-567812345678")
        assert result.rendered_prompt == "Test prompt"
        assert result.dream_discord_message_id == 987654321
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_audit_by_dream_message_not_found(self) -> None:
        """Test retrieving non-existent audit returns None."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = _create_mock_pool(mock_conn)

        result = await get_audit_by_dream_message(
            db_pool=mock_pool, dream_discord_message_id=999999999
        )

        assert result is None


class TestGetAuditsWithFeedback:
    """Tests for get_audits_with_feedback function."""

    @pytest.mark.asyncio
    async def test_get_audits_with_feedback_found(self) -> None:
        """Test retrieving audits with feedback."""
        mock_conn = MagicMock()
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
                "context_config": {"episodic": 5, "semantic": 3, "graph": 0},
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
                "context_config": {"episodic": 3, "semantic": 2, "graph": 1},
                "archetype_matched": None,
                "archetype_confidence": None,
                "reasoning": None,
                "main_discord_message_id": 333,
                "dream_discord_message_id": 444,
                "feedback_id": UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
                "created_at": datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            },
        ]

        mock_rows = []
        for data in audit_data_list:
            mock_row = MagicMock()
            mock_row.__getitem__.side_effect = lambda key, d=data: d[key]
            mock_rows.append(mock_row)

        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = _create_mock_pool(mock_conn)

        result = await get_audits_with_feedback(db_pool=mock_pool, limit=10)

        assert len(result) == 2
        assert result[0].audit_id == UUID("11111111-1111-1111-1111-111111111111")
        assert result[1].audit_id == UUID("22222222-2222-2222-2222-222222222222")
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_audits_with_feedback_empty(self) -> None:
        """Test retrieving audits when none have feedback."""
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = _create_mock_pool(mock_conn)

        result = await get_audits_with_feedback(db_pool=mock_pool, limit=10)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_audits_with_feedback_custom_limit(self) -> None:
        """Test retrieving audits with custom limit."""
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = _create_mock_pool(mock_conn)

        result = await get_audits_with_feedback(db_pool=mock_pool, limit=5)

        assert len(result) == 0
        call_args = mock_conn.fetch.call_args
        assert call_args[0][1] == 5
        assert call_args[0][2] == 0
