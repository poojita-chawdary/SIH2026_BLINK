# ==============================================================================
# PROJECT BLINK: Zero-Payload Rapid Scanning Engine for INSAT-3DS/3DR
# Edge Inference & Post-Processing Container
# ==============================================================================

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libhdf5-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set default python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code and assets
COPY config/ ./config/
COPY src/ ./src/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY README.md .

# Create data directories
RUN mkdir -p data/raw_netcdf data/processed_tensors

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Launch FastAPI Server
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
