FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for eBPF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    linux-headers-generic \
    libbpf-dev \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (but agent needs elevated caps)
RUN useradd --create-home --shell /bin/bash appuser

# Create directories
RUN mkdir -p /etc/security_agent/rules /var/lib/security_agent/quarantine \
    && chown -R appuser:appuser /var/lib/security_agent

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY src/ src/
COPY tests/ tests/

# Set ownership
RUN chown -R appuser:appuser /app

# Note: Container needs --privileged or specific capabilities for eBPF
# CAP_SYS_ADMIN, CAP_BPF, CAP_PERFMON

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import security_agent; print('healthy')" || exit 1

# Expose metrics port
EXPOSE 8005

CMD ["python", "-m", "security_agent"]
