"""Unit tests for config module."""

import os
from unittest.mock import patch

import pytest

from lattice.utils.config import Config, _get_env_bool, _get_env_int, get_config


class TestGetEnvBool:
    """Tests for _get_env_bool helper function."""

    def test_returns_true_for_true_string(self) -> None:
        """Test that 'true' string returns True."""
        with patch.dict(os.environ, {"TEST_VAR": "true"}):
            assert _get_env_bool("TEST_VAR") is True

    def test_returns_true_for_1_string(self) -> None:
        """Test that '1' string returns True."""
        with patch.dict(os.environ, {"TEST_VAR": "1"}):
            assert _get_env_bool("TEST_VAR") is True

    def test_returns_true_for_yes_string(self) -> None:
        """Test that 'yes' string returns True."""
        with patch.dict(os.environ, {"TEST_VAR": "yes"}):
            assert _get_env_bool("TEST_VAR") is True

    def test_returns_true_for_on_string(self) -> None:
        """Test that 'on' string returns True."""
        with patch.dict(os.environ, {"TEST_VAR": "on"}):
            assert _get_env_bool("TEST_VAR") is True

    def test_returns_false_for_false_string(self) -> None:
        """Test that 'false' string returns False."""
        with patch.dict(os.environ, {"TEST_VAR": "false"}):
            assert _get_env_bool("TEST_VAR") is False

    def test_returns_false_for_0_string(self) -> None:
        """Test that '0' string returns False."""
        with patch.dict(os.environ, {"TEST_VAR": "0"}):
            assert _get_env_bool("TEST_VAR") is False

    def test_returns_false_for_no_string(self) -> None:
        """Test that 'no' string returns False."""
        with patch.dict(os.environ, {"TEST_VAR": "no"}):
            assert _get_env_bool("TEST_VAR") is False

    def test_returns_false_for_off_string(self) -> None:
        """Test that 'off' string returns False."""
        with patch.dict(os.environ, {"TEST_VAR": "off"}):
            assert _get_env_bool("TEST_VAR") is False

    def test_returns_default_when_not_set(self) -> None:
        """Test that default value is returned when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_bool("NONEXISTENT", default=True) is True
            assert _get_env_bool("NONEXISTENT", default=False) is False

    def test_case_insensitive(self) -> None:
        """Test that parsing is case-insensitive."""
        with patch.dict(os.environ, {"TEST_VAR": "TRUE"}):
            assert _get_env_bool("TEST_VAR") is True
        with patch.dict(os.environ, {"TEST_VAR": "FALSE"}):
            assert _get_env_bool("TEST_VAR") is False


class TestGetEnvInt:
    """Tests for _get_env_int helper function."""

    def test_parses_positive_integer(self) -> None:
        """Test parsing a positive integer."""
        with patch.dict(os.environ, {"TEST_VAR": "42"}):
            assert _get_env_int("TEST_VAR") == 42

    def test_parses_negative_integer(self) -> None:
        """Test parsing a negative integer."""
        with patch.dict(os.environ, {"TEST_VAR": "-10"}):
            assert _get_env_int("TEST_VAR") == -10

    def test_parses_zero(self) -> None:
        """Test parsing zero."""
        with patch.dict(os.environ, {"TEST_VAR": "0"}):
            assert _get_env_int("TEST_VAR") == 0

    def test_returns_default_when_not_set(self) -> None:
        """Test that default value is returned when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_int("NONEXISTENT", default=100) == 100
            assert _get_env_int("NONEXISTENT", default=0) == 0

    def test_returns_default_on_invalid_integer(self) -> None:
        """Test that default value is returned for invalid integer strings."""
        with patch.dict(os.environ, {"TEST_VAR": "not_a_number"}):
            assert _get_env_int("TEST_VAR", default=10) == 10

    def test_returns_default_on_float_string(self) -> None:
        """Test that default value is returned for float strings."""
        with patch.dict(os.environ, {"TEST_VAR": "3.14"}):
            assert _get_env_int("TEST_VAR", default=10) == 10


class TestConfigLoad:
    """Tests for Config.load() method."""

    def test_loads_all_discord_settings(self) -> None:
        """Test that all Discord settings are loaded correctly."""
        env_vars = {
            "DISCORD_TOKEN": "test_token_123",
            "DISCORD_MAIN_CHANNEL_ID": "123456",
            "DISCORD_DREAM_CHANNEL_ID": "789012",
            "DISCORD_GUILD_ID": "guild_123",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            assert config.discord_token == "test_token_123"
            assert config.discord_main_channel_id == 123456
            assert config.discord_dream_channel_id == 789012
            assert config.discord_guild_id == "guild_123"

    def test_loads_all_database_settings(self) -> None:
        """Test that all database settings are loaded correctly."""
        env_vars = {
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "DB_POOL_MIN_SIZE": "3",
            "DB_POOL_MAX_SIZE": "10",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            assert config.database_url == "postgresql://user:pass@localhost/db"
            assert config.db_pool_min_size == 3
            assert config.db_pool_max_size == 10

    def test_loads_all_llm_settings(self) -> None:
        """Test that all LLM settings are loaded correctly."""
        env_vars = {
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test_api_key",
            "OPENROUTER_MODEL": "anthropic/claude-3.5-sonnet",
            "OPENROUTER_TIMEOUT": "60",
            "GEMINI_API_KEY": "gemini_key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            assert config.llm_provider == "openrouter"
            assert config.openrouter_api_key == "test_api_key"
            assert config.openrouter_model == "anthropic/claude-3.5-sonnet"
            assert config.openrouter_timeout == 60
            assert config.gemini_api_key == "gemini_key"

    def test_loads_all_application_settings(self) -> None:
        """Test that all application settings are loaded correctly."""
        env_vars = {
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "DEBUG",
            "STRUCTURED_LOGS": "false",
            "HEALTH_PORT": "9090",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            assert config.environment == "development"
            assert config.log_level == "DEBUG"
            assert config.structured_logs is False
            assert config.health_port == 9090

    def test_uses_defaults_when_not_set(self) -> None:
        """Test that default values are used when env vars not set."""
        with patch("lattice.utils.config.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load()

                assert config.discord_token is None
                assert config.discord_main_channel_id == 0
                assert config.discord_dream_channel_id == 0
                assert config.discord_guild_id is None
                assert config.database_url is None
                assert config.db_pool_min_size == 2
                assert config.db_pool_max_size == 5
                assert config.llm_provider == "placeholder"
                assert config.openrouter_api_key is None
                assert config.openrouter_model == "anthropic/claude-3.5-sonnet"
                assert config.openrouter_timeout == 30
                assert config.gemini_api_key is None
                assert config.environment == "production"
                assert config.log_level == "INFO"
                assert config.structured_logs is True
                assert config.health_port == 8080

    def test_log_level_uppercased(self) -> None:
        """Test that log level is automatically uppercased."""
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}, clear=True):
            config = Config.load()
            assert config.log_level == "DEBUG"

    def test_handles_missing_channel_ids(self) -> None:
        """Test that channel IDs default to 0 when not set or invalid."""
        env_vars = {
            "DISCORD_MAIN_CHANNEL_ID": "invalid",
            "DISCORD_DREAM_CHANNEL_ID": "also_invalid",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            assert config.discord_main_channel_id == 0
            assert config.discord_dream_channel_id == 0


class TestGetConfig:
    """Tests for get_config() function."""

    def test_returns_singleton_instance(self) -> None:
        """Test that get_config returns the same instance on multiple calls."""
        with patch.dict(os.environ, {"ENVIRONMENT": "test1"}, clear=True):
            config1 = get_config()
            config2 = get_config()

            assert config1 is config2

    def test_reload_updates_existing_instance(self) -> None:
        """Test that reload=True updates the existing instance."""
        # Reset global config
        import lattice.utils.config as config_module

        config_module._config = None

        with patch.dict(os.environ, {"ENVIRONMENT": "initial"}, clear=True):
            config1 = get_config()
            assert config1.environment == "initial"

        # Change environment variable and reload
        with patch.dict(os.environ, {"ENVIRONMENT": "updated"}, clear=True):
            config2 = get_config(reload=True)

            # Should be the same object
            assert config1 is config2
            # But with updated values
            assert config2.environment == "updated"
            assert config1.environment == "updated"  # First reference also updated

        # Clean up
        config_module._config = None

    def test_creates_instance_on_first_call(self) -> None:
        """Test that get_config creates an instance on first call."""
        # Reset global config
        import lattice.utils.config as config_module

        config_module._config = None

        with patch.dict(os.environ, {"ENVIRONMENT": "first_call"}, clear=True):
            config = get_config()

            assert config.environment == "first_call"
            assert config_module._config is not None

        # Clean up
        config_module._config = None

    def test_does_not_reload_without_flag(self) -> None:
        """Test that config is not reloaded when reload=False."""
        # Reset global config
        import lattice.utils.config as config_module

        config_module._config = None

        with patch.dict(os.environ, {"ENVIRONMENT": "original"}, clear=True):
            config1 = get_config()
            assert config1.environment == "original"

        # Change environment variable but don't reload
        with patch.dict(os.environ, {"ENVIRONMENT": "changed"}, clear=True):
            config2 = get_config(reload=False)

            # Should still have original value
            assert config2.environment == "original"
            assert config1 is config2

        # Clean up
        config_module._config = None


class TestConfigEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handles_empty_strings(self) -> None:
        """Test that empty strings are handled correctly."""
        env_vars = {
            "DISCORD_TOKEN": "",
            "DATABASE_URL": "",
            "LLM_PROVIDER": "",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            # Empty strings should be preserved as empty strings, not None
            assert config.discord_token == ""
            assert config.database_url == ""
            assert config.llm_provider == ""

    def test_handles_whitespace_strings(self) -> None:
        """Test that whitespace-only strings are preserved."""
        with patch.dict(os.environ, {"DISCORD_TOKEN": "   "}, clear=True):
            config = Config.load()

            # Whitespace should be preserved
            assert config.discord_token == "   "

    def test_handles_very_large_integers(self) -> None:
        """Test handling of very large integer values."""
        env_vars = {
            "DISCORD_MAIN_CHANNEL_ID": "9999999999999999",
            "DB_POOL_MAX_SIZE": "1000000",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            assert config.discord_main_channel_id == 9999999999999999
            assert config.db_pool_max_size == 1000000

    def test_handles_special_characters_in_strings(self) -> None:
        """Test that special characters in strings are preserved."""
        env_vars = {
            "DISCORD_TOKEN": "token_with_!@#$%^&*()_special_chars",
            "DATABASE_URL": "postgresql://user:p@ss!w0rd@localhost/db?ssl=true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config.load()

            assert config.discord_token == "token_with_!@#$%^&*()_special_chars"
            assert (
                config.database_url
                == "postgresql://user:p@ss!w0rd@localhost/db?ssl=true"
            )


class TestConfigDataclass:
    """Tests for Config dataclass properties."""

    def test_config_is_dataclass(self) -> None:
        """Test that Config is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(Config)

    def test_config_has_all_required_fields(self) -> None:
        """Test that Config has all expected fields."""
        from dataclasses import fields

        config = Config.load()
        field_names = {f.name for f in fields(config)}

        expected_fields = {
            # Discord
            "discord_token",
            "discord_main_channel_id",
            "discord_dream_channel_id",
            "discord_guild_id",
            # Database
            "database_url",
            "db_pool_min_size",
            "db_pool_max_size",
            # LLM
            "llm_provider",
            "openrouter_api_key",
            "openrouter_model",
            "openrouter_timeout",
            "gemini_api_key",
            # Application
            "environment",
            "log_level",
            "structured_logs",
            "health_port",
        }

        assert field_names == expected_fields
