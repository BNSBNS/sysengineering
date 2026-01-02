FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DB_ENGINE_DATA_DIR=/data \
    DB_ENGINE_WAL_DIR=/wal

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Create data and WAL directories
RUN mkdir -p /data /wal

# Expose ports
# 5432: PostgreSQL-compatible wire protocol
# 8001: Prometheus metrics
# 50051: gRPC API
EXPOSE 5432 8001 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.connect(('localhost', 8001)); s.close()"

# Run the database engine
CMD ["python", "-m", "db_engine.main"]
