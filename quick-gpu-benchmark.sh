#!/bin/bash

# Quick GPU Benchmark - One-liner installer and runner
# Usage: curl -sSL https://raw.githubusercontent.com/cameron-huber/gpu-benchmark/main/quick-gpu-benchmark.sh | bash -s -- [OPTIONS]
# Or: wget -qO- https://raw.githubusercontent.com/cameron-huber/gpu-benchmark/main/quick-gpu-benchmark.sh | bash -s -- [OPTIONS]

set -e

REPO_URL="https://github.com/cameron-huber/gpu-benchmark.git"
BENCHMARK_DIR="gpu-benchmark-temp"
COST_PER_HOUR="1.00"
OUTPUT_FILE=""
VERBOSE=false
CLEANUP=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cost-per-hour)
            COST_PER_HOUR="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --keep-files)
            CLEANUP=false
            shift
            ;;
        --help)
            echo "Quick GPU Benchmark Runner"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cost-per-hour COST    Set GPU cost per hour (default: 1.00)"
            echo "  --output-file FILE      Save results to file"
            echo "  --verbose              Show detailed output"
            echo "  --keep-files           Don't cleanup temporary files"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Quick GPU Benchmark Runner"
echo "==============================="

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "📥 Downloading benchmark suite..."
git clone --depth 1 "$REPO_URL" "$BENCHMARK_DIR" >/dev/null 2>&1

cd "$BENCHMARK_DIR"

echo "🔧 Setting up environment..."
chmod +x setup.sh
./setup.sh >/dev/null 2>&1

echo "🎯 Running GPU benchmark..."
echo "   Cost per hour: $COST_PER_HOUR"

# Build command
CMD="source venv/bin/activate && python gpu_benchmark.py --cost-per-hour $COST_PER_HOUR"

if [[ -n "$OUTPUT_FILE" ]]; then
    CMD="$CMD --output-file $OUTPUT_FILE"
    echo "   Output file: $OUTPUT_FILE"
fi

if [[ "$VERBOSE" == "true" ]]; then
    CMD="$CMD --verbose"
fi

echo ""

# Run the benchmark
bash -c "$CMD"

# Copy output file to original directory if specified
if [[ -n "$OUTPUT_FILE" && -f "$OUTPUT_FILE" ]]; then
    cp "$OUTPUT_FILE" "$OLDPWD/"
    echo ""
    echo "📄 Results saved to: $OLDPWD/$OUTPUT_FILE"
fi

# Cleanup
if [[ "$CLEANUP" == "true" ]]; then
    cd /
    rm -rf "$TEMP_DIR"
    echo "🧹 Cleaned up temporary files"
else
    echo "📁 Temporary files kept at: $TEMP_DIR/$BENCHMARK_DIR"
fi

echo "✅ Benchmark complete!"
