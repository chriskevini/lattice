"""Unit tests for lattice/utils/config.py."""

from unittest.mock import patch

import pytest

from lattice.utils.config import (
    Config,
    _get_env_bool,
    _get_env_int,
    get_config,
)


class TestGetEnvBool:
    """Tests for _get_env_bool helper function."""

    def test_returns_default_when_env_var_not_set(self) -> None:
        """Test that default value is returned when env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            result = _get_env_bool("MISSING_VAR")
            assert result is False

    def test_returns_default_when_env_var_is_empty(self) -> None:
        """Test that default is returned when env var is empty string."""
        with patch.dict("os.environ", {"EMPTY_VAR": ""}):
            result = _get_env_bool("EMPTY_VAR")
            assert result is False

    def test_parses_true_variants(self) -> None:
        """Test that various 'true' values are correctly parsed."""
        true_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
        for val in true_values:
            with patch.dict("os.environ", {"TEST_VAR": val}):
                result = _get_env_bool("TEST_VAR")
                assert result is True, f"Failed for value: {val}"

    def test_parses_false_variants(self) -> None:
        """Test that various 'false' values are correctly parsed."""
        false_values = ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF", ""]
        for val in false_values:
            with patch.dict("os.environ", {"TEST_VAR": val}):
                result = _get_env_bool("TEST_VAR")
                assert result is False, f"Failed for value: {val}"

    def test_custom_default(self) -> None:
        """Test that custom default value is used when env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            result = _get_env_bool("MISSING_VAR", default=True)
            assert result is True


class TestGetEnvInt:
    """Tests for _get_env_int helper function."""

    def test_returns_default_when_env_var_not_set(self) -> None:
        """Test that default value is returned when env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            result = _get_env_int("MISSING_VAR")
            assert result == 0

    def test_returns_default_when_env_var_is_empty(self) -> None:
        """Test that default is returned when env var is empty string."""
        with patch.dict("os.environ", {"EMPTY_VAR": ""}):
            result = _get_env_int("EMPTY_VAR")
            assert result == 0

    def test_parses_valid_integer(self) -> None:
        """Test that valid integer strings are correctly parsed."""
        test_cases = [
            ("42", 42),
            ("0", 0),
            ("-10", -10),
            ("123456", 123456),
        ]
        for env_val, expected in test_cases:
            with patch.dict("os.environ", {"TEST_VAR": env_val}):
                result = _get_env_int("TEST_VAR")
                assert result == expected, f"Failed for value: {env_val}"

    def test_returns_default_for_invalid_integer(self) -> None:
        """Test that default is returned for non-integer strings."""
        invalid_values = ["abc", "12.5", "true", "not-a-number", "1a2b"]
        for val in invalid_values:
            with patch.dict("os.environ", {"TEST_VAR": val}):
                result = _get_env_int("TEST_VAR")
                assert result == 0, f"Failed for value: {val}"

    def test_custom_default(self) -> None:
        """Test that custom default value is used when env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            result = _get_env_int("MISSING_VAR", default=99)
            assert result == 99

    def test_custom_default_for_invalid_integer(self) -> None:
        """Test that custom default is used for invalid integer strings."""
        with patch.dict("os.environ", {"TEST_VAR": "invalid"}):
            result = _get_env_int("TEST_VAR", default=42)
            assert result == 42


class TestConfigLoad:
    """Tests for Config.load() class method."""

    def setup_method(self) -> None:
        """Reset global config before each test."""
        import lattice.utils.config

        lattice.utils.config._config = None

    @pytest.fixture
    def clean_env(self) -> dict[str, str]:
        """Provide a clean environment for testing."""
        return {
            "DISCORD_TOKEN": "test-token",
            "DISCORD_MAIN_CHANNEL_ID": "12345",
            "DISCORD_DREAM_CHANNEL_ID": "67890",
            "DISCORD_GUILD_ID": "99999",
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "DB_POOL_MIN_SIZE": "3",
            "DB_POOL_MAX_SIZE": "10",
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-api-key",
            "OPENROUTER_MODEL": "test/model",
            "OPENROUTER_TIMEOUT": "60",
            "GEMINI_API_KEY": "gemini-key",
            "ENVIRONMENT": "development",
            "LOG_LEVEL": "DEBUG",
            "STRUCTURED_LOGS": "false",
            "HEALTH_PORT": "9090",
        }

    def test_load_with_all_env_vars(self, clean_env: dict[str, str]) -> None:
        """Test Config.load() with all environment variables set."""
        with patch.dict("os.environ", clean_env, clear=True):
            config = Config.load()

            assert config.discord_token == "test-token"
            assert config.discord_main_channel_id == 12345
            assert config.discord_dream_channel_id == 67890
            assert config.discord_guild_id == "99999"
            assert config.database_url == "postgresql://user:pass@localhost/db"
            assert config.db_pool_min_size == 3
            assert config.db_pool_max_size == 10
            assert config.llm_provider == "openrouter"
            assert config.openrouter_api_key == "test-api-key"
            assert config.openrouter_model == "test/model"
            assert config.openrouter_timeout == 60
            assert config.gemini_api_key == "gemini-key"
            assert config.environment == "development"
            assert config.log_level == "DEBUG"
            assert config.structured_logs is False
            assert config.health_port == 9090

    def test_load_with_minimal_env_vars(self) -> None:
        """Test Config.load() with only required environment variables."""
        import lattice.utils.config

        lattice.utils.config._config = None
        env = {
            "DISCORD_TOKEN": "minimal-token",
            "DISCORD_MAIN_CHANNEL_ID": "100",
            "DISCORD_DREAM_CHANNEL_ID": "200",
            "DATABASE_URL": "postgresql://localhost/test",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lattice.utils.config.load_dotenv"):
                config = Config.load()

                assert config.discord_token == "minimal-token"
                assert config.discord_main_channel_id == 100
                assert config.discord_dream_channel_id == 200
                assert config.database_url == "postgresql://localhost/test"
                # Defaults for optional fields
                assert config.db_pool_min_size == 2
                assert config.db_pool_max_size == 5
                assert config.llm_provider == "placeholder"
                assert config.openrouter_model == "anthropic/claude-3.5-sonnet"
                assert config.openrouter_timeout == 30
                assert config.environment == "production"
                assert config.log_level == "INFO"
                assert config.structured_logs is True
                assert config.health_port == 8080

    def test_load_with_missing_optional_env_vars(self) -> None:
        """Test Config.load() when optional env vars are missing."""
        import lattice.utils.config

        lattice.utils.config._config = None
        env = {
            "DISCORD_TOKEN": "token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lattice.utils.config.load_dotenv"):
                config = Config.load()

                assert config.discord_guild_id is None
                assert config.openrouter_api_key is None
                assert config.gemini_api_key is None

    def test_load_parses_boolean_correctly(self) -> None:
        """Test that boolean environment variables are correctly parsed."""
        import lattice.utils.config

        lattice.utils.config._config = None
        env = {
            "DISCORD_TOKEN": "token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
            "STRUCTURED_LOGS": "true",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lattice.utils.config.load_dotenv"):
                config = Config.load()
                assert config.structured_logs is True

        env["STRUCTURED_LOGS"] = "false"
        with patch.dict("os.environ", env, clear=True):
            with patch("lattice.utils.config.load_dotenv"):
                config = Config.load()
                assert config.structured_logs is False

    def test_load_parses_integer_correctly(self) -> None:
        """Test that integer environment variables are correctly parsed."""
        import lattice.utils.config

        lattice.utils.config._config = None
        env = {
            "DISCORD_TOKEN": "token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
            "DB_POOL_MIN_SIZE": "5",
            "DB_POOL_MAX_SIZE": "20",
            "OPENROUTER_TIMEOUT": "120",
            "HEALTH_PORT": "3000",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lattice.utils.config.load_dotenv"):
                config = Config.load()

                assert config.db_pool_min_size == 5
                assert config.db_pool_max_size == 20
                assert config.openrouter_timeout == 120
                assert config.health_port == 3000

    def test_load_handles_invalid_integer_gracefully(self) -> None:
        """Test that invalid integer env vars fall back to defaults."""
        import lattice.utils.config

        lattice.utils.config._config = None
        env = {
            "DISCORD_TOKEN": "token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
            "DB_POOL_MIN_SIZE": "invalid",
            "DB_POOL_MAX_SIZE": "also-invalid",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lattice.utils.config.load_dotenv"):
                config = Config.load()

                # Should fall back to defaults
                assert config.db_pool_min_size == 2
                assert config.db_pool_max_size == 5


class TestGetConfig:
    """Tests for get_config() function."""

    def setup_method(self) -> None:
        """Reset global config before each test."""
        import lattice.utils.config

        lattice.utils.config._config = None

    def test_get_config_returns_singleton(self) -> None:
        """Test that get_config returns the same instance on subsequent calls."""
        env = {
            "DISCORD_TOKEN": "singleton-token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }
        with patch.dict("os.environ", env, clear=True):
            config1 = get_config()
            config2 = get_config()

            assert config1 is config2

    def test_get_config_reload_flag_forces_reload(self) -> None:
        """Test that reload=True forces config to be reloaded."""
        env1 = {
            "DISCORD_TOKEN": "first-token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }
        env2 = {
            "DISCORD_TOKEN": "second-token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }

        with patch.dict("os.environ", env1, clear=True):
            config1 = get_config()
            assert config1.discord_token == "first-token"

        with patch.dict("os.environ", env2, clear=True):
            config2 = get_config(reload=True)
            assert config2.discord_token == "second-token"

    def test_get_config_without_reload_uses_cached(self) -> None:
        """Test that without reload flag, cached config is returned."""
        env1 = {
            "DISCORD_TOKEN": "cached-token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }
        env2 = {
            "DISCORD_TOKEN": "new-token",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }

        with patch.dict("os.environ", env1, clear=True):
            _config1 = get_config()

        # Change environment but don't reload
        with patch.dict("os.environ", env2, clear=True):
            config2 = get_config(reload=False)
            # Should still have the original token
            assert config2.discord_token == "cached-token"

    def test_get_config_updates_in_place_on_reload(self) -> None:
        """Test that reload updates the existing instance in place."""
        env1 = {
            "DISCORD_TOKEN": "original",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }
        env2 = {
            "DISCORD_TOKEN": "updated",
            "DISCORD_MAIN_CHANNEL_ID": "1",
            "DISCORD_DREAM_CHANNEL_ID": "2",
            "DATABASE_URL": "postgresql://localhost/db",
        }

        with patch.dict("os.environ", env1, clear=True):
            config1 = get_config()
            original_id = id(config1)

        with patch.dict("os.environ", env2, clear=True):
            config2 = get_config(reload=True)
            # Should be the same object
            assert id(config2) == original_id
            assert config2.discord_token == "updated"


class TestConfigDataclass:
    """Tests for Config as a dataclass."""

    def test_config_is_dataclass(self) -> None:
        """Test that Config is a dataclass."""
        assert hasattr(Config, "__dataclass_fields__")

    def test_config_fields_exist(self) -> None:
        """Test that all expected fields exist."""
        fields = {f.name for f in Config.__dataclass_fields__.values()}
        expected_fields = {
            "discord_token",
            "discord_main_channel_id",
            "discord_dream_channel_id",
            "discord_guild_id",
            "database_url",
            "db_pool_min_size",
            "db_pool_max_size",
            "llm_provider",
            "openrouter_api_key",
            "openrouter_model",
            "openrouter_timeout",
            "gemini_api_key",
            "environment",
            "log_level",
            "structured_logs",
            "health_port",
        }
        assert fields == expected_fields

    def test_config_field_types(self) -> None:
        """Test that Config fields have correct types."""
        fields = Config.__dataclass_fields__
        assert fields["discord_token"].type == str | None  # noqa: E721
        assert fields["discord_main_channel_id"].type == int  # noqa: E721
        assert fields["discord_dream_channel_id"].type == int  # noqa: E721
        assert fields["discord_guild_id"].type == str | None  # noqa: E721
        assert fields["database_url"].type == str | None  # noqa: E721
        assert fields["db_pool_min_size"].type == int  # noqa: E721
        assert fields["db_pool_max_size"].type == int  # noqa: E721
        assert fields["llm_provider"].type == str  # noqa: E721
        assert fields["openrouter_api_key"].type == str | None  # noqa: E721
        assert fields["openrouter_model"].type == str  # noqa: E721
        assert fields["openrouter_timeout"].type == int  # noqa: E721
        assert fields["gemini_api_key"].type == str | None  # noqa: E721
        assert fields["environment"].type == str  # noqa: E721
        assert fields["log_level"].type == str  # noqa: E721
        assert fields["structured_logs"].type == bool  # noqa: E721
        assert fields["health_port"].type == int  # noqa: E721

    def test_config_default_values_through_load(self) -> None:
        """Test that defaults work correctly through load()."""
        import lattice.utils.config

        lattice.utils.config._config = None
        env = {
            "DISCORD_MAIN_CHANNEL_ID": "0",
            "DISCORD_DREAM_CHANNEL_ID": "0",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lattice.utils.config.load_dotenv"):
                config = Config.load()

                assert config.llm_provider == "placeholder"
                assert config.openrouter_model == "anthropic/claude-3.5-sonnet"
                assert config.openrouter_timeout == 30
                assert config.environment == "production"
                assert config.log_level == "INFO"
                assert config.structured_logs is True
                assert config.health_port == 8080
                assert config.db_pool_min_size == 2
                assert config.db_pool_max_size == 5
