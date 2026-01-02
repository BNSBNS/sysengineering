FROM nvidia/cuda:12.2-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[dev]"

# Copy application code
COPY src/ src/
COPY tests/ tests/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import gpu_platform; print('healthy')" || exit 1

# Expose metrics port
EXPOSE 8003

CMD ["python3", "-m", "gpu_platform"]
