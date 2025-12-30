"""Unit tests for Lattice memory modules."""

from lattice.memory.episodic import EpisodicMessage
from lattice.memory.procedural import PromptTemplate
from lattice.memory.semantic import StableFact


class TestEpisodicMemory:
    """Tests for episodic memory module."""

    def test_episodic_message_creation(self) -> None:
        """Test that we can create an EpisodicMessage object."""
        message = EpisodicMessage(
            content="Hello world",
            discord_message_id=123456789,
            channel_id=987654321,
            is_bot=False,
        )

        assert message.content == "Hello world"
        assert message.discord_message_id == 123456789
        assert message.channel_id == 987654321
        assert message.is_bot is False


class TestSemanticMemory:
    """Tests for semantic memory module."""

    def test_stable_fact_creation(self) -> None:
        """Test that we can create a StableFact object."""
        fact = StableFact(
            content="User prefers dark mode",
            entity_type="preference",
        )

        assert fact.content == "User prefers dark mode"
        assert fact.entity_type == "preference"


class TestProceduralMemory:
    """Tests for procedural memory module."""

    def test_prompt_template_creation(self) -> None:
        """Test that we can create a PromptTemplate object."""
        template = PromptTemplate(
            prompt_key="TEST_PROMPT",
            template="Hello {name}",
            temperature=0.7,
        )

        assert template.prompt_key == "TEST_PROMPT"
        assert template.template == "Hello {name}"
        assert template.temperature == 0.7
