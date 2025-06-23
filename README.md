# GPU Benchmarking Suite

This repository provides a comprehensive GPU benchmarking script to evaluate hardware performance, ML framework capabilities, and container/orchestration compatibility across multiple environments. It integrates synthetic tests, real-model inference, data-loader benchmarks, multi‑GPU scaling, and cost metrics.

## Features

* **System Setup**: GPU & CPU details, memory, driver/CUDA/cuDNN/cuBLAS versions, PCIe, power draw.
* **Hardware & Performance**: Disk I/O, network/ping, uptime, power efficiency, InfiniBand, NVLink.
* **ML Framework Tests**:

  * Framework & precision detection (PyTorch, TF, JAX; FP32/FP16).
  * Synthetic training benchmark (MLP on random data).
  * Hugging Face inference (DistilBERT sentiment, GPT-2 medium generative):

    * Cold-start time, throughput, cache growth, model size, memory footprint.
  * Proxy Transformer training (DistilBERT classification) with mean/median/min/max throughput.
  * Data Loader benchmark (CIFAR-10 with augmentation) reporting sample/token throughput & cost stats.
  * Cost metrics: per-sample & per-token (\$), based on `--cost_per_hour`.
  * Determinism check.
* **Multi-GPU & Parallelism**:

  * True DDP training (multi-process) throughput.
  * NCCL all-reduce latency & P2P bandwidth sweeps (1 MB, 16 MB, 128 MB).
  * Multi-GPU inference scaling.
  * Model parallelism placeholder.
* **Container & Orchestration Compatibility**:

  * Docker & kubectl installation & version.
  * GPU visibility inside a test container.
  * Optional environment-matrix mode (`--env_matrix`) to run across multiple CUDA/cuDNN Docker images.

## Requirements

* **OS**: Linux with NVIDIA drivers
* **Python**: 3.6+
* **Dependencies**:

  ```bash
  pip install torch torchvision transformers kubernetes
  ```
* **Tools**: Docker, kubectl, nvidia-smi, ping, ip, awk, lscpu

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

## Usage

1. **Single-run**: benchmark current environment and cost metrics:

   ```bash
   python gpu_benchmark.py --cost_per_hour 3.50
   ```
2. **Environment matrix**: test across multiple CUDA/cuDNN stacks:

   ```bash
   python gpu_benchmark.py --env_matrix --cost_per_hour 3.50
   ```

## Output

* Live logs for each test with descriptive headers.
* Final summary tables for each category:

  * *System Setup*
  * *Hardware & Performance*
  * *ML Framework Tests*
  * *Multi-GPU & Parallelism*
  * *Container & Orchestration Compatibility*

## Customization

* **Cost per hour**: adjust `--cost_per_hour` to match instance pricing.
* **Data-loader**: change dataset, batch size, augmentations in `data_loader_benchmark()`.
* **Proxy/Generative models**: modify model names or prompt lengths.
* **Multi-GPU**: set `world_size` or use `torchrun` for multi-node scenarios.
* **Env matrix**: update `images` list in `compare_envs()` for supported CUDA versions.

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
## Sample Output

=== Summary: System Setup ===
+----------------------+----------------------+
| Metric               | Value                |
+----------------------+----------------------+
| GPU Model            | NVIDIA A100-SXM4     |
| CPU Model            | Intel(R) Xeon(R)...  |
| Total RAM            | 512.00 GB            |
| VRAM Capacity        | 40.00 GB             |
| VRAM Usage           | 20.00 GB             |
| CUDA Available       | True                 |
| Driver Version       | 535.54               |
| CUDA Version         | 12.1                 |
| NCCL Version         | 2.18                 |
| cuDNN Version        | 8.8                  |
| cuBLAS Version       | 11.15                |
| PCIe Version         | PCIe4.0              |
| Power Draw           | 240 W                |
+----------------------+----------------------+

=== Summary: Hardware & Performance ===
+--------------------------------+------------------+
| Metric                         | Value            |
+--------------------------------+------------------+
| Disk Write Speed               | 980.12 MB/s      |
| Disk Read Speed                | 1123.47 MB/s     |
| Network Bandwidth              | 1.00 Gbps        |
| Ping / Latency                 | 0.85 ms          |
| Uptime                         | 24.25 hrs        |
| Power Efficiency               | 11.72 TFLOPS/W   |
| InfiniBand RDMA Bandwidth      | 90.00 Gbps       |
| NVLink Link 0 State            | Enabled          |
| NVLink Link 0 Max Speed        | 25.78 GB/s       |
| NVLink Link 0 Utilization      | 45 %             |
+--------------------------------+------------------+

=== Summary: ML Framework Tests ===
+---------------------------------------------+------------------------+
| Metric                                      | Value                  |
+---------------------------------------------+------------------------+
| Framework                                   | PyTorch                |
| Precision Level                             | FP16                   |
| Training Throughput                         | 480.25 samples/sec     |
| Time per Epoch                              | 66.00 s                |
| GPU-Hours to Convergence                    | 0.0183 GPU-hrs         |
| Gradient Sync Time                          | 0.00 ms                |
| Final Loss                                  | 0.3125                 |
| Final Accuracy                              | 90.50 %                |
| Memory Footprint (training)                 | 1.95 GB                |
| Cold-Start Time                             | 40.00 ms               |
| Model Size                                  | 0.35 GB                |
| Memory Footprint (inference)                | 3.20 GB                |
| HF Inference Latency                        | 78.00 ms               |
| HF Throughput                               | 12.82 requests/sec     |
| Cost / 1M Tokens Trained                    | $0.00396               |
| Cost / Inference                            | $0.000076              |
| Proxy Model Throughput (mean)               | 320.45 samples/sec     |
| Proxy Model Throughput (std)                | 5.12                   |
| Proxy Model Throughput (min)                | 310.00 samples/sec     |
| Proxy Model Throughput (max)                | 330.75 samples/sec     |
| Gen[10] Latency                             | 45.00 ms               |
| Gen[10] Throughput                          | 22.22 req/sec          |
| Gen[10] Cache Growth                        | 150.00 MB              |
| Gen[50] Latency                             | 75.00 ms               |
| Gen[50] Throughput                          | 13.33 req/sec          |
| Gen[50] Cache Growth                        | 200.00 MB              |
| Gen[100] Latency                            | 110.00 ms              |
| Gen[100] Throughput                         | 9.09 req/sec           |
| Gen[100] Cache Growth                       | 250.00 MB              |
| Gen[200] Latency                            | 180.00 ms              |
| Gen[200] Throughput                         | 5.56 req/sec           |
| Gen[200] Cache Growth                       | 350.00 MB              |
| Sample Throughput (mean)                    | 850.00 per_sec         |
| Sample Throughput (median)                  | 845.00 per_sec         |
| Sample Throughput (min)                     | 830.00 per_sec         |
| Sample Throughput (max)                     | 860.00 per_sec         |
| Token Throughput (mean)                     | 34600000.00 per_sec    |
| Token Throughput (median)                   | 34400000.00 per_sec    |
| Token Throughput (min)                      | 33800000.00 per_sec    |
| Token Throughput (max)                      | 35200000.00 per_sec    |
| Cost per Sample (mean)                      | $0.000012              |
| Cost per Sample (median)                    | $0.000012              |
| Cost per Sample (min)                       | $0.000010              |
| Cost per Sample (max)                       | $0.000013              |
| Cost per Token (mean)                       | $0.000000              |
| Cost per Token (median)                     | $0.000000              |
| Cost per Token (min)                        | $0.000000              |
| Cost per Token (max)                        | $0.000000              |
| Determinism Check                           | True                   |
+---------------------------------------------+------------------------+

=== Summary: Multi-GPU & Parallelism
Container & Orchestration Compatibility ===
+--------------------------------------+---------------------+
| Metric                               | Value               |
+--------------------------------------+---------------------+
| Kubernetes Support                   | False               |
| Orchestration Tooling                | kubectl             |
| Multi-GPU Throughput                 | 850.00 samples/sec  |
| All-Reduce Latency                   | 12.00us             |
| P2P Bandwidth                        | 9000MB/s            |
| Multi-GPU Inf Throughput             | 25.00 req/sec       |
| Model Parallelism                    | N/A                 |
| Docker Installed                     | True                |
| Docker Version                       | Docker version 20.10.12, build e91ed57 |
| Kubernetes CLI Installed             | True                |
| Kubernetes CLI Version               | Client Version: v1.27.3 |
| GPU Visibility in Container          | True                |
+--------------------------------------+---------------------+
