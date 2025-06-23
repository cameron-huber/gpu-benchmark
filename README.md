# GPU Benchmarking Tool

A standalone Python script to perform one-line, end-to-end benchmarking of GPU instances. It collects system, library, compute, storage, network, training, and inference metrics and prints them in a single ASCII table.

---

## FEATURES

- **System & GPU Info**: utilization, VRAM, PCIe version, power draw, driver & CUDA versions  
- **Library Versions**: cuDNN, NCCL, cuBLAS  
- **Storage I/O**: sequential read/write throughput  
- **Network**: interface bandwidth & ping latency  
- **Uptime**: system up-time in hours  
- **Framework Detection**: PyTorch, TensorFlow, JAX and default precision  
- **Orchestration**: Kubernetes support & tooling presence  
- **Training Benchmark**: synthetic MNIST-style loop (throughput, epoch time, loss, accuracy, GPU-hrs, memory footprint)  
- **Inference Benchmark**: model latency, throughput, cold-start time, variance, model size  
- **Power Efficiency**: TFLOPS/W measured via matrix-multiply  
- **Determinism**: bit-wise repeatability check  
- **Infra Checks**: InfiniBand bandwidth, NVLink (if available)  
- **Placeholders** for any custom cost or MTTR metrics  

---

## PREREQUISITES

- **Python** 3.6+  
- **NVIDIA GPU** with drivers & CUDA toolkit  
- **PyTorch** installed:  
  ```bash
  pip install torch
  ```
- System tools in your `PATH`:  
  - `nvidia-smi`  
  - `ping`  
  - `ip`, `awk`  

---

## INSTALL

### Quick Setup (Recommended)

```bash
git clone https://github.com/<YOUR_USER>/gpu-benchmark.git
cd gpu-benchmark
./setup.sh
```

This will:
- Create a virtual environment
- Install all Python dependencies
- Check system requirements
- Verify GPU setup

### Manual Install

```bash
git clone https://github.com/<YOUR_USER>/gpu-benchmark.git
cd gpu-benchmark
pip install -r requirements.txt
chmod +x gpu_benchmark.py
```

### Using Make (Alternative)

```bash
git clone https://github.com/<YOUR_USER>/gpu-benchmark.git
cd gpu-benchmark
make setup
```

---

## USING THE TOOL

### One-line via `curl`

Fetch and execute the latest script:

```bash
curl -s https://raw.githubusercontent.com/<YOUR_USER>/gpu-benchmark/main/gpu_benchmark.py \
  | python3 -
```

[Download the raw script ↗](https://raw.githubusercontent.com/<YOUR_USER>/gpu-benchmark/main/gpu_benchmark.py)

### Using the Run Script (Recommended)

```bash
# Basic benchmark with default cost ($0.50/hour)
./run.sh

# Specify custom cost per hour
./run.sh --cost_per_hour 1.20

# Run environment matrix tests
./run.sh --cost_per_hour 0.80 --env_matrix
```

### Using Make Commands

```bash
# Run with default settings
make run

# Run with custom cost
make run COST=1.50

# Run environment matrix tests
make test

# View all available commands
make help
```

### Or run directly

```bash
python3 gpu_benchmark.py --cost_per_hour 0.50
```

### Docker Usage

```bash
# Build the Docker image
make docker-build

# Run in container
make docker-run COST=0.75

# Or manually
docker build -t gpu-benchmark .
docker run --gpus all --rm gpu-benchmark --cost_per_hour 0.50
```

---

## SAMPLE OUTPUT

```text
+----------------------------------------------+------------------+
| Metric                                       | Value            |
+----------------------------------------------+------------------+
| GPU Utilization                              | 85 %             |
| VRAM Usage                                   | 23.5 GB          |
| CUDA Available                               | True             |
| Disk Write Speed                             | 1234.56 MB/s     |
| Disk Read Speed                              | 1450.78 MB/s     |
| Network Bandwidth                            | 10.0 Gbps        |
| Ping / Latency                               | 0.72 ms          |
| Power Draw                                   | 250 W            |
| Training Throughput                          | 512.00 samples/s |
| Time per Epoch                               | 120.00 s         |
| Final Loss                                   | 0.0450           |
| Final Accuracy                               | 98.20 %          |
| GPU-Hours to Convergence                     | 0.0333 GPU-hrs   |
| Gradient Sync Time                           | 0.00 ms          |
| Inference Latency                            | 1.50 ms          |
| Inference Throughput                         | 6667.00 requests/sec |
| Cold-Start Time                              | 35.00 ms         |
| Model Size                                   | 1.20 GB          |
| Memory Footprint                             | 0.05 GB          |
| Power Efficiency                             | 12.30 TFLOPS/W   |
| Determinism Check                            | True             |
| Uptime                                       | 72.00 hrs        |
| Error Rate                                   | 0 errors/hr      |
| Throughput Std. Dev.                         | 2.50 %           |
| Framework                                    | PyTorch          |
| Precision Level                              | FP16             |
| Driver Version                               | 535.54           |
| CUDA Version                                 | 12.1             |
| NCCL Version                                 | 2.18             |
| cuDNN Version                                | 8.8              |
| cuBLAS Version                               | 11.15            |
| Kubernetes Support                           | True             |
| Orchestration Tooling                        | kubectl,airflow  |
| PCIe Version                                 | PCIe4.0          |
| NVLink Bandwidth                             | N/A              |
| InfiniBand Bandwidth                         | 200 Gbps         |
| Fault Tolerance / MTTR                       | N/A              |
+----------------------------------------------+------------------+
```
