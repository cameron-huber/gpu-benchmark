#!/bin/bash

# GPU Benchmark Tool Runner Script
# This script activates the virtual environment and runs the benchmark

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    GPU Benchmark Tool                        ║"
    echo "║              Comprehensive GPU Performance Testing           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check if setup has been run
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

if [ ! -f "venv/bin/activate" ]; then
    print_error "Virtual environment seems corrupted. Please run ./setup.sh again."
    exit 1
fi

# Show banner
print_banner

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
python3 -c "import torch, transformers, numpy" 2>/dev/null || {
    print_error "Required packages not found. Please run ./setup.sh first."
    exit 1
}

# Default cost per hour if not provided
DEFAULT_COST=0.50

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 --cost_per_hour COST [--env_matrix] [other options]"
    echo ""
    echo "Examples:"
    echo "  $0 --cost_per_hour 0.50                    # Basic benchmark"
    echo "  $0 --cost_per_hour 1.20 --env_matrix       # Multi-environment test"
    echo ""
    echo "Using default cost: \$${DEFAULT_COST}/hour"
    echo ""
    ARGS="--cost_per_hour $DEFAULT_COST"
else
    ARGS="$@"
fi

# Show system info before running
print_status "System Information:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi
echo ""

# Run the benchmark
print_status "Starting GPU benchmark..."
python3 gpu_benchmark.py $ARGS

# Check exit code
if [ $? -eq 0 ]; then
    print_status "✅ Benchmark completed successfully!"
else
    print_error "❌ Benchmark failed with errors."
    exit 1
fi
