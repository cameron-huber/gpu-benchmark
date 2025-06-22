# GPU Benchmarking Tool

A standalone Python script to perform one-line, end-to-end benchmarking of GPU instances. It collects system, library, compute, storage, network, training, and inference metrics and prints them in a single ASCII table.

---

## Features

- **System & GPU Info**: utilization, VRAM, PCIe version, power draw, driver & CUDA versions  
- **Library Versions**: cuDNN, NCCL, cuBLAS  
- **Storage I/O**: sequential read/write throughput  
- **Network**: interface bandwidth & ping latency  
- **Uptime**: system up-time in hours  
- **Framework Detection**: PyTorch, TensorFlow, JAX and default precision  
- **Orchestration**: Kubernetes support & tooling presence  
- **Training Benchmark**: synthetic MNIST-style loop (throughput, epoch time, loss, accuracy, GPU-hrs, memory footprint)  
- **Inference Benchmark**: model latency, throughput, cold-start time, variance, model size  
- **Power Efficiency**: TFLOPS/W measured via matrix‐multiply  
- **Determinism**: bit-wise repeatability check  
- **Infra Checks**: InfiniBand bandwidth, NVLink (if available)  
- **Placeholders** for any custom cost or MTTR metrics  

---

## Prerequisites

- Python 3.6+  
- NVIDIA GPU with drivers & CUDA toolkit  
- PyTorch installed:  
  ```bash
  pip install torch

---

## Installation

git clone https://github.com/<YOUR_USER>/gpu-benchmark.git
cd gpu-benchmark
chmod +x gpu_benchmark.py

---

**One-line via `curl`:**

```bash
curl -s https://raw.githubusercontent.com/<YOUR_USER>/gpu-benchmark/main/gpu_benchmark.py \
  | python3 -


Or Locally:
python3 gpu_benchmark.py

---
## 🖥️ Example Output

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
