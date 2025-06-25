#!/usr/bin/env python3
"""
GPU Benchmarking Script with Hugging Face Integration

Usage:
    python gpu_benchmark.py [--cost_per_hour COST] [--env_matrix]

Requirements:
    - Python 3.6+
    - NVIDIA GPU with drivers, CUDA toolkit
    - PyTorch (pip install torch)
    - Transformers (pip install transformers)
    - torchvision (pip install torchvision)
    - Kubernetes client (pip install kubernetes)
    - Docker, kubectl, nvidia-smi, ping, ip, awk, lscpu in PATH
"""
import argparse
import subprocess, sys, time, os, ctypes
from shutil import which
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        return None

def print_header(title):
    print(f"\n=== {title} ===")

def print_table(rows):
    if not rows:
        print("(no data)")
        return
    header = ('Metric','Value')
    all_rows = [header] + rows
    w1 = max(len(r[0]) for r in all_rows)
    w2 = max(len(r[1]) for r in all_rows)
    sep = '+' + '-'*(w1+2) + '+' + '-'*(w2+2) + '+'
    print(sep)
    print(f"| {header[0].ljust(w1)} | {header[1].ljust(w2)} |")
    print(sep)
    for m,v in rows:
        print(f"| {m.ljust(w1)} | {v.ljust(w2)} |")
    print(sep)

# (Previous functions: gpu_system_metrics, cpu_ram_metrics, lib_versions,
#  disk_io, network_ping, uptime, framework_precision, k8s_tooling,
#  training_metrics, huggingface_inference, power_efficiency,
#  determinism_check, ib_bandwidth, nvlink_bandwidth,
#  multi_gpu_training_metrics, gradient_sync_latency, cross_gpu_bandwidth,
#  multi_gpu_inference, model_parallelism_scaling,
#  proxy_model_training, generative_model_benchmark)

# Data Loader Benchmark
def data_loader_benchmark(rows, dataset='CIFAR10', batch_size=64, num_workers=4, num_batches=50):
    print_header("Data Loader Benchmark")
    print(f"-> Loading {dataset} for {num_batches} batches with batch_size={batch_size}")
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    try:
        ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    except:
        rows.append(("Data Loader", "Dataset load failed"))
        return
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    timings, tokens_per_batch = [], []
    for i, (x,_) in enumerate(dl):
        if i >= num_batches: break
        t0 = time.time()
        _ = x.cuda()
        torch.cuda.synchronize()
        timings.append(time.time()-t0)
        tokens_per_batch.append(x.numel())
    rates_samples = np.array([batch_size/t for t in timings])
    rates_tokens  = np.array([tokens_per_batch[i]/timings[i] for i in range(len(timings))])
    for name, arr in [('Sample Throughput', rates_samples), ('Token Throughput', rates_tokens)]:
        rows.append((f"{name} (mean)",   f"{arr.mean():.2f} per_sec"))
        rows.append((f"{name} (median)", f"{np.median(arr):.2f} per_sec"))
        rows.append((f"{name} (min)",    f"{arr.min():.2f} per_sec"))
        rows.append((f"{name} (max)",    f"{arr.max():.2f} per_sec"))
    # cost arrays
    cost_unit = COST_PER_HOUR/3600
    cost_samples = cost_unit/rates_samples
    cost_tokens  = cost_unit/rates_tokens
    for name, arr in [('Cost per Sample', cost_samples), ('Cost per Token', cost_tokens)]:
        rows.append((f"{name} (mean)",   f"${arr.mean():.6f}"))
        rows.append((f"{name} (median)", f"${np.median(arr):.6f}"))
        rows.append((f"{name} (min)",    f"${arr.min():.6f}"))
        rows.append((f"{name} (max)",    f"${arr.max():.6f}"))

# Container & Orchestration Compatibility
def container_checks(rows):
    print_header("Container & Orchestration Compatibility")
    # Docker
    docker_path = which('docker')
    if docker_path:
        ver = run_cmd('docker --version') or 'N/A'
        rows.append(("Docker Installed", "True"))
        rows.append(("Docker Version", ver))
    else:
        rows.append(("Docker Installed", "False"))
        rows.append(("Docker Version", "N/A"))
    # kubectl
    kubectl_path = which('kubectl')
    if kubectl_path:
        ver = run_cmd('kubectl version --client --short') or 'N/A'
        rows.append(("Kubernetes CLI Installed", "True"))
        rows.append(("Kubernetes CLI Version", ver))
    else:
        rows.append(("Kubernetes CLI Installed", "False"))
        rows.append(("Kubernetes CLI Version", "N/A"))
    # GPU visibility
    gpu_vis = run_cmd('docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi --query-gpu=name --format=csv,noheader')
    rows.append(("GPU Visibility in Container", "True" if gpu_vis else "False"))

# Environment matrix runner
def compare_envs(images):
    print_header("Environment Matrix Benchmarking")
    for img in images:
        print_header(f"Container Image: {img}")
        cmd = (
            f"docker run --gpus all --rm -v {os.getcwd()}:/app -w /app "
            f"--entrypoint python {img} gpu_benchmark.py --skip-env-matrix --cost_per_hour {COST_PER_HOUR}"
        )
        out = run_cmd(cmd)
        print(out or f"[failed in {img}]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost_per_hour',    type=float, required=True, help="GPU $/hour")
    parser.add_argument('--env_matrix',       action='store_true', help="Run across multiple CUDA/container environments")
    parser.add_argument('--skip-env-matrix', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    COST_PER_HOUR = args.cost_per_hour

    if args.env_matrix and not args.skip_env_matrix:
        images = [
            'nvidia/cuda:11.8-cudnn8-runtime',
            'nvidia/cuda:12.1-cudnn8-runtime',
            'nvidia/cuda:12.1-cudnn8-devel',
            'nvidia/cuda:11.8-cudnn8-devel'
        ]
        compare_envs(images)
        sys.exit(0)

    categories = {
        'System Setup': [],
        'Hardware & Performance': [],
        'ML Framework Tests': [],
        'Multi-GPU & Parallelism\nContainer & Orchestration Compatibility': []
    }

    # 1. System Setup
    gpu_system_metrics(categories['System Setup'])
    cpu_ram_metrics(categories['System Setup'])
    lib_versions(categories['System Setup'])

    # 2. Hardware & Performance
    disk_io(categories['Hardware & Performance'])
    network_ping(categories['Hardware & Performance'])
    uptime(categories['Hardware & Performance'])
    power_efficiency(categories['Hardware & Performance'])
    ib_bandwidth(categories['Hardware & Performance'])
    nvlink_bandwidth(categories['Hardware & Performance'])

    # 3. ML Framework Tests
    framework_precision(categories['ML Framework Tests'])
    training_metrics(categories['ML Framework Tests'])
    huggingface_inference(categories['ML Framework Tests'])
    proxy_model_training(categories['ML Framework Tests'])
    generative_model_benchmark(categories['ML Framework Tests'])
    data_loader_benchmark(categories['ML Framework Tests'])
    determinism_check(categories['ML Framework Tests'])

    # 4. Multi-GPU & Parallelism + Container Compatibility
    k8s_tooling(categories['Multi-GPU & Parallelism\nContainer & Orchestration Compatibility'])
    multi_gpu_training_metrics(categories['Multi-GPU & Parallelism\nContainer & Orchestration Compatibility'])
    gradient_sync_latency(categories['Multi-GPU & Parallelism\nContainer & Orchestration Compatibility'])
    cross_gpu_bandwidth(categories['Multi-GPU & Parallelism\nContainer & Orchestration Compatibility'])
    multi_gpu_inference(categories['Multi-GPU & Parallelism\nContainer & Orchestration Compatibility'])
    model_parallelism_scaling(categories['Multi-GPU & Parallelism\nContainer & Orchestration Compatibility'])
    container_checks(categories['Multi-GPU & Parallelism\nContainer & Orchestration Compatibility'])

    # Print summaries
    for title, rows in categories.items():
        print_header(f"Summary: {title}")
        print_table(rows)


def gpu_system_metrics(category):
    category.append("GPU System Metrics Placeholder")

def cpu_ram_metrics(category):
    category.append("CPU & RAM Metrics Placeholder")

def lib_versions(category):
    category.append("Library Versions Placeholder")

def disk_io(category):
    category.append("Disk I/O Placeholder")

def network_ping(category):
    category.append("Network Ping Placeholder")

def uptime(category):
    category.append("System Uptime Placeholder")

def power_efficiency(category):
    category.append("Power Efficiency Placeholder")

def ib_bandwidth(category):
    category.append("Infiniband Bandwidth Placeholder")

def nvlink_bandwidth(category):
    category.append("NVLink Bandwidth Placeholder")

def framework_precision(category):
    category.append("Framework Precision Placeholder")

def training_metrics(category):
    category.append("Training Metrics Placeholder")

def huggingface_inference(category):
    category.append("Hugging Face Inference Placeholder")

def proxy_model_training(category):
    category.append("Proxy Model Training Placeholder")

def generative_model_benchmark(category):
    category.append("Generative Model Benchmark Placeholder")

def determinism_check(category):
    category.append("Determinism Check Placeholder")

def k8s_tooling(category):
    category.append("K8s Tooling Placeholder")

def multi_gpu_training_metrics(category):
    category.append("Multi-GPU Training Metrics Placeholder")

def gradient_sync_latency(category):
    category.append("Gradient Sync Latency Placeholder")

def cross_gpu_bandwidth(category):
    category.append("Cross-GPU Bandwidth Placeholder")

def multi_gpu_inference(category):
    category.append("Multi-GPU Inference Placeholder")

def model_parallelism_scaling(category):
    category.append("Model Parallelism Scaling Placeholder")
