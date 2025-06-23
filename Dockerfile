# GPU Benchmark Tool Docker Image
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    git \
    iputils-ping \
    iproute2 \
    gawk \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the benchmark script
COPY gpu_benchmark.py .

# Make script executable
RUN chmod +x gpu_benchmark.py

# Set default entrypoint
ENTRYPOINT ["python3", "gpu_benchmark.py"]

# Default command with placeholder for cost
CMD ["--cost_per_hour", "0.50"]

# Labels for metadata
LABEL maintainer="GPU Benchmark Tool"
LABEL description="Containerized GPU benchmarking tool for comprehensive performance testing"
LABEL version="1.0"
