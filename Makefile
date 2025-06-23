.PHONY: help setup install run clean test lint docker-build docker-run

# Default cost per hour for benchmarking
COST ?= 0.50

# Help target - shows available commands
help:
	@echo "GPU Benchmark Tool - Available Commands:"
	@echo ""
	@echo "  make setup        - Install dependencies and set up environment"
	@echo "  make run          - Run benchmark with default settings ($$$(COST)/hour)"
	@echo "  make run COST=1.2 - Run benchmark with custom cost per hour"
	@echo "  make test         - Run environment tests"
	@echo "  make clean        - Clean up temporary files and virtual environment"
	@echo "  make lint         - Check code formatting (requires flake8)"
	@echo "  make docker-build - Build Docker image for benchmarking"
	@echo "  make docker-run   - Run benchmark in Docker container"
	@echo ""
	@echo "Examples:"
	@echo "  make setup && make run"
	@echo "  make run COST=2.50"
	@echo "  make test"

# Setup environment and install dependencies
setup:
	@echo "🚀 Setting up GPU Benchmark Tool..."
	@./setup.sh

# Alias for setup
install: setup

# Run the benchmark with specified cost
run:
	@./run.sh --cost_per_hour $(COST)

# Run with environment matrix testing
test:
	@./run.sh --cost_per_hour $(COST) --env_matrix

# Clean up virtual environment and temporary files
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf venv/
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@rm -rf data/  # CIFAR10 dataset downloaded by script
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete"

# Lint the code (optional - requires flake8)
lint:
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "🔍 Linting code..."; \
		flake8 gpu_benchmark.py --max-line-length=120; \
	else \
		echo "⚠️  flake8 not found. Install with: pip install flake8"; \
	fi

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	@docker build -t gpu-benchmark .

docker-run:
	@echo "🐳 Running benchmark in Docker..."
	@docker run --gpus all --rm gpu-benchmark --cost_per_hour $(COST)

# Quick system check
check:
	@echo "🔍 System Check:"
	@echo "Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@echo "NVIDIA-SMI: $$(nvidia-smi --version 2>/dev/null | head -1 || echo 'Not found')"
	@echo "CUDA: $$(nvcc --version 2>/dev/null | grep 'release' || echo 'Not found')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not found')"
