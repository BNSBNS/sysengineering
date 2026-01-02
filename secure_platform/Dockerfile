FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY src/ ./src/
COPY tests/ ./tests/

RUN mkdir -p /certs /policies /audit

EXPOSE 8443 8003

HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import socket; s = socket.socket(); s.connect(('localhost', 8003)); s.close()"

CMD ["python", "-m", "secure_platform.main"]
