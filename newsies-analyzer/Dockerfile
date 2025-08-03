# Dockerfile for Newsies Analyzer Service
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false

# Install system dependencies for NLP processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy source code for dependencies first
COPY newsies-common ./newsies-common
COPY newsies-clients ./newsies-clients
COPY newsies-scraper ./newsies-scraper
COPY newsies-trainer ./newsies-trainer
COPY newsies-analyzer ./newsies-analyzer

# Install packages in correct order
RUN pip install -e ./newsies-common && \
    pip install -e ./newsies-clients && \
    pip install -e ./newsies-scraper && \
    pip install -e ./newsies-trainer && \
    pip install -e ./newsies-analyzer

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import newsies_analyzer; print('OK')" || exit 1

# Run the analyzer service
CMD ["python", "-m", "newsies_analyzer.main"]
