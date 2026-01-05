# Multi-stage build for optimal image size
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code
COPY pyproject.toml uv.lock* README.md ./
COPY lattice/ ./lattice/
COPY scripts/ ./scripts/

# Install UV and sync dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev && \
    cp -r .venv /opt/venv && \
    rm -rf .venv /root/.cache/uv

ENV PATH="/opt/venv/bin:$PATH"

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY lattice/ ./lattice/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Create directories for logs
RUN mkdir -p /app/logs

# Set Python to run in unbuffered mode (better for logs)
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import asyncio; asyncio.run(__import__('asyncpg').connect('${DATABASE_URL}'))" || exit 1

# Run the bot
CMD ["python", "-m", "lattice"]
