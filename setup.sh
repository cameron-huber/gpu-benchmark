#!/bin/bash

# GPU Benchmark Tool Setup Script
# This script sets up the environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up GPU Benchmark Tool..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.6+ is available
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.6+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Found Python $PYTHON_VERSION"

# Check if NVIDIA GPU and drivers are available
print_status "Checking NVIDIA GPU setup..."
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. Make sure NVIDIA drivers are installed."
else
    print_status "nvidia-smi found:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_status "Found CUDA version: $CUDA_VERSION"
else
    print_warning "CUDA toolkit not found in PATH. Some features may not work."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
print_status "Installing PyTorch..."
if command -v nvcc &> /dev/null; then
    # Install PyTorch with CUDA support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    # Install CPU-only version
    print_warning "Installing CPU-only PyTorch (no CUDA detected)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
print_status "Installing other dependencies..."
pip install -r requirements.txt

# Check system tools
print_status "Checking system tools..."
MISSING_TOOLS=()

for tool in ping ip awk lscpu; do
    if ! command -v $tool &> /dev/null; then
        MISSING_TOOLS+=($tool)
    fi
done

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    print_warning "Missing system tools: ${MISSING_TOOLS[*]}"
    print_warning "Some benchmarks may not work properly."
fi

# Check optional tools
print_status "Checking optional tools..."
for tool in docker kubectl; do
    if command -v $tool &> /dev/null; then
        print_status "$tool is available"
    else
        print_warning "$tool not found (optional for container/k8s features)"
    fi
done

# Make the benchmark script executable
chmod +x gpu_benchmark.py

print_status "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the benchmark: ./run.sh --cost_per_hour 0.50"
echo "3. Or run directly: python3 gpu_benchmark.py --cost_per_hour 0.50"
echo ""
echo "📖 For more options, see: python3 gpu_benchmark.py --help"
