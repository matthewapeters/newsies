# Dockerfile for Newsies Trainer Service (GPU-enabled)
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

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

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install packages in correct order
RUN pip install -e ./newsies-common && \
    pip install -e ./newsies-clients && \
    pip install -e ./newsies-scraper && \
    pip install -e ./newsies-trainer

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import newsies_trainer; print('OK')" || exit 1

# Run the trainer service
CMD ["python", "-m", "newsies_trainer.main"]
