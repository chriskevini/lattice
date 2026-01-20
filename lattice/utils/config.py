"""Centralized environment variable access and configuration management."""

import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Helper to parse boolean environment variables."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int = 0) -> int:
    """Helper to parse integer environment variables."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass
class Config:
    """Lattice application configuration."""

    # Discord Settings
    discord_token: str | None
    discord_main_channel_id: int
    discord_dream_channel_id: int
    discord_guild_id: str | None

    # Database Settings
    database_url: str | None
    db_pool_min_size: int
    db_pool_max_size: int

    # LLM Settings
    llm_provider: str
    openrouter_api_key: str | None
    openrouter_model: str
    openrouter_timeout: int
    gemini_api_key: str | None

    # Embedding Settings
    embedding_model: str
    embedding_dimension: int
    embedding_provider: str

    # Memory Settings
    enable_triple_memory: bool
    enable_embedding_memory: bool
    embedding_similarity_threshold: float

    # Application Settings
    environment: str
    log_level: str
    structured_logs: bool
    health_port: int

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables."""
        # Ensure .env is loaded
        load_dotenv()

        return cls(
            discord_token=os.getenv("DISCORD_TOKEN"),
            discord_main_channel_id=_get_env_int("DISCORD_MAIN_CHANNEL_ID", 0),
            discord_dream_channel_id=_get_env_int("DISCORD_DREAM_CHANNEL_ID", 0),
            discord_guild_id=os.getenv("DISCORD_GUILD_ID"),
            database_url=os.getenv("DATABASE_URL"),
            db_pool_min_size=_get_env_int("DB_POOL_MIN_SIZE", 2),
            db_pool_max_size=_get_env_int("DB_POOL_MAX_SIZE", 5),
            llm_provider=os.getenv("LLM_PROVIDER", "placeholder"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"
            ),
            openrouter_timeout=_get_env_int("OPENROUTER_TIMEOUT", 30),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimension=_get_env_int("EMBEDDING_DIMENSION", 384),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openrouter"),
            enable_triple_memory=_get_env_bool("ENABLE_TRIPLE_MEMORY", True),
            enable_embedding_memory=_get_env_bool("ENABLE_EMBEDDING_MEMORY", False),
            embedding_similarity_threshold=float(
                os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.7")
            ),
            environment=os.getenv("ENVIRONMENT", "production"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            structured_logs=_get_env_bool("STRUCTURED_LOGS", True),
            health_port=_get_env_int("HEALTH_PORT", 8080),
        )


# Global configuration instance
_config: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Get the global configuration instance.

    Args:
        reload: If True, force reload the configuration from environment variables.
    """
    global _config
    if _config is None:
        _config = Config.load()
    elif reload:
        new_config = Config.load()
        # Update existing instance in place so all module imports see changes
        for field in fields(_config):
            setattr(_config, field.name, getattr(new_config, field.name))
    return _config


# For backward compatibility and easy access
config = get_config()
