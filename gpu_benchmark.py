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
    - Kubernetes client (pip install kubernetes)
    - torchvision for data loader benchmark (pip install torchvision)
    - Docker, kubectl, nvidia-smi, ping, ip, awk, lscpu in PATH
"""
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
    sep = f'+-{{:-<{w1}}}-+-{{:-<{w2}}}-+'.format('','')
    print(sep)
    print(f"| {header[0].ljust(w1)} | {header[1].ljust(w2)} |")
    print(sep)
    for m,v in rows:
        print(f"| {m.ljust(w1)} | {v.ljust(w2)} |")
    print(sep)

# 1. System Setup
# ... previous implementations omitted for brevity ...
# 2. Hardware & Performance
# ... omitted ...
# 3. ML Framework Tests
# ... proxy_model_training(), generative_model_benchmark(), etc.

# Data Loader Benchmark
def data_loader_benchmark(rows, dataset='CIFAR10', batch_size=64, num_workers=4, num_batches=50):
    print_header("Data Loader Benchmark")
    print(f"-> Loading {dataset} for {num_batches} batches with batch_size={batch_size}")
    # select dataset
    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    try:
        ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    except:
        rows.append(("Data Loader", "Dataset load failed"))
        return
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    timings, tokens_per_batch = [], []
    for i, (x,_) in enumerate(dl):
        if i>=num_batches: break
        t0 = time.time()
        _ = x.cuda()
        torch.cuda.synchronize()
        timings.append(time.time()-t0)
        tokens_per_batch.append(x.numel())
    rates_samples = np.array([batch_size/t for t in timings])
    rates_tokens = np.array([tokens_per_batch[i]/timings[i] for i in range(len(timings))])
    for name, arr in [('Sample Throughput', rates_samples), ('Token Throughput', rates_tokens)]:
        rows.append((f"{name} (mean)", f"{arr.mean():.2f} per_sec"))
        rows.append((f"{name} (median)", f"{np.median(arr):.2f} per_sec"))
        rows.append((f"{name} (min)", f"{arr.min():.2f} per_sec"))
        rows.append((f"{name} (max)", f"{arr.max():.2f} per_sec"))
    # cost
    cost = COST_PER_HOUR/3600
    cost_samples = cost/rates_samples
    cost_tokens = cost/rates_tokens
    for name, arr in [('Cost per Sample', cost_samples), ('Cost per Token', cost_tokens)]:
        rows.append((f"{name} (mean)", f"${arr.mean():.6f}"))
        rows.append((f"{name} (median)", f"${np.median(arr):.6f}"))
        rows.append((f"{name} (min)", f"${arr.min():.6f}"))
        rows.append((f"{name} (max)", f"${arr.max():.6f}"))

# main integration omitted for brevity

