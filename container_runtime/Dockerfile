FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install system dependencies for container runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    iptables \
    bridge-utils \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY src/ ./src/
COPY tests/ ./tests/

# Create required directories
RUN mkdir -p /var/lib/containers /run/containers

EXPOSE 50052 8002

# Requires privileged mode for namespace operations
CMD ["python", "-m", "container_runtime.main"]
