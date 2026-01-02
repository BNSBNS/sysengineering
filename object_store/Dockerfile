FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Create data directories
RUN mkdir -p /var/lib/object_store/data /var/lib/object_store/metadata \
    && chown -R appuser:appuser /var/lib/object_store

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY src/ src/
COPY tests/ tests/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Volumes for persistent storage
VOLUME ["/var/lib/object_store/data", "/var/lib/object_store/metadata"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import object_store; print('healthy')" || exit 1

# Expose ports: S3 API, metrics
EXPOSE 9000 8006

CMD ["python", "-m", "object_store"]
