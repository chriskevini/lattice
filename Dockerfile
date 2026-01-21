# Multi-stage build for optimal image size
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files only (not source code - will be copied later)
COPY pyproject.toml uv.lock* README.md ./

# Install UV and sync dependencies to a temporary location
RUN pip install --no-cache-dir --disable-pip-version-check uv && \
    uv sync --no-dev --frozen && \
    cp -r .venv /opt/venv && \
    rm -rf .venv /root/.cache/uv /root/.cache/pip

ENV PATH="/opt/venv/bin:$PATH"

# Production stage
FROM python:3.12-slim

# Install only minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Copy application code (after venv to leverage layer caching)
COPY lattice/ ./lattice/
COPY scripts/ ./scripts/
COPY avatars/ ./avatars/
COPY pyproject.toml ./

# Create directories for logs
RUN mkdir -p /app/logs && \
    chmod 755 /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import asyncio; asyncio.run(__import__('asyncpg').connect('${DATABASE_URL}'))" || exit 1

# Run bot
CMD ["python", "-m", "lattice"]
