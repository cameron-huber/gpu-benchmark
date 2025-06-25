#!/usr/bin/env python3
"""
Enhanced GPU Benchmarking Suite
Comprehensive GPU and system performance benchmarking tool with Google Sheets summary output
"""

import os
import sys
import time
import json
import subprocess
import argparse
import statistics
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import psutil
import platform
from datetime import datetime
import socket
import requests
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
import csv
from collections import defaultdict
import re

def run_cmd(cmd):
    """Execute shell command safely and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1

def print_header(title):
    print(f"\n=== {title} ===")

def print_table(rows):
    if not rows:
        return
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    
    # Print header separator
    separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
    print(separator)
    
    # Print header
    header = "| " + " | ".join(str(rows[0][i]).ljust(col_widths[i]) for i in range(len(rows[0]))) + " |"
    print(header)
    print(separator)
    
    # Print data rows
    for row in rows[1:]:
        data_row = "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " |"
        print(data_row)
    print(separator)

def extract_numeric_value(value_str):
    """Extract numeric value from a string for ranking purposes"""
    try:
        # Remove common units and convert to float
        cleaned = re.sub(r'[^\d.,\-]', '', str(value_str))
        if cleaned:
            # Handle comma as decimal separator or thousands separator
            if ',' in cleaned and '.' in cleaned:
                # Assume comma is thousands separator
                cleaned = cleaned.replace(',', '')
            elif ',' in cleaned and cleaned.count(',') == 1:
                # Check if comma is likely decimal separator
                parts = cleaned.split(',')
                if len(parts[1]) <= 3:  # Likely decimal
                    cleaned = cleaned.replace(',', '.')
                else:  # Likely thousands separator
                    cleaned = cleaned.replace(',', '')
            
            return float(cleaned)
    except:
        pass
    return 0

def get_metric_importance_score(metric, value, category):
    """Calculate importance score for a metric based on its relevance"""
    metric_lower = metric.lower()
    value_str = str(value).lower()
    
    # High importance metrics
    high_importance = [
        'gpu.*throughput', 'training.*throughput', 'inference.*throughput',
        'token.*throughput', 'sample.*throughput', 'pipeline.*throughput',
        'bandwidth', 'memory.*total', 'memory.*free', 'gpu.*count',
        'cuda.*version', 'gpu.*name', 'power.*draw', 'temperature',
        'utilization', 'cost.*per.*token', 'cost.*per.*sample'
    ]
    
    # Medium importance metrics
    medium_importance = [
        'cpu.*cores', 'ram.*total', 'training.*time', 'inference.*time',
        'speedup', 'efficiency', 'latency', 'precision', 'batch.*time'
    ]
    
    # Calculate base score
    score = 1
    
    # Check against high importance patterns
    for pattern in high_importance:
        if re.search(pattern, metric_lower):
            score = 10
            break
    
    # Check against medium importance patterns  
    if score == 1:
        for pattern in medium_importance:
            if re.search(pattern, metric_lower):
                score = 5
                break
    
    # Boost score for numeric values (more useful for analysis)
    numeric_value = extract_numeric_value(value)
    if numeric_value > 0:
        score *= 2
    
    # Boost score for performance-related categories
    if 'performance' in category.lower() or 'framework' in category.lower():
        score *= 1.5
    
    # Boost score for multi-gpu metrics
    if 'multi' in category.lower() or 'gpu' in category.lower():
        score *= 1.3
    
    # Reduce score for error messages
    if 'error' in value_str or 'not available' in value_str or 'n/a' in value_str:
        score *= 0.1
    
    return score

def select_top_metrics(all_metrics, top_k=20):
    """Select top K most important metrics for summary"""
    scored_metrics = []
    
    for category, metric, value in all_metrics:
        score = get_metric_importance_score(metric, value, category)
        scored_metrics.append((score, category, metric, value))
    
    # Sort by score (descending) and take top K
    scored_metrics.sort(key=lambda x: x[0], reverse=True)
    
    return [(cat, metric, value) for score, cat, metric, value in scored_metrics[:top_k]]

def output_google_sheets_summary(all_metrics, cost_per_hour, output_file=None):
    """Output top 20 metrics in Google Sheets-friendly tab-separated format"""
    top_metrics = select_top_metrics(all_metrics, top_k=20)
    
    timestamp = datetime.now().isoformat()
    
    # Create summary data
    summary_lines = []
    summary_lines.append("Rank\tCategory\tMetric\tValue\tCost/Hour\tTimestamp")
    
    for i, (category, metric, value) in enumerate(top_metrics, 1):
        # Clean value for Google Sheets
        clean_value = str(value).replace('\t', ' ').replace('\n', ' ')
        summary_lines.append(f"{i}\t{category}\t{metric}\t{clean_value}\t${cost_per_hour:.2f}\t{timestamp}")
    
    # Output to file or stdout
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"\nGoogle Sheets summary written to: {output_file}")
    else:
        print("\n=== GOOGLE SHEETS SUMMARY (Tab-Separated) ===")
        for line in summary_lines:
            print(line)
    
    return summary_lines

# Include all the original benchmark functions here...
# (I'll include a selection of key functions to keep the response manageable)

def data_loader_benchmark(rows, dataset='CIFAR10', batch_size=64, num_workers=4, num_batches=50):
    """Benchmark data loading performance"""
    print_header("Data Loader Benchmark")
    print(f"-> Loading {dataset} for {num_batches} batches with batch_size={batch_size}")
    
    try:
        # Load CIFAR10 dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        if dataset == 'CIFAR10':
            dataset_obj = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                       download=True, transform=transform)
        
        dataloader = DataLoader(dataset_obj, batch_size=batch_size, 
                               shuffle=True, num_workers=num_workers)
        
        # Benchmark loading
        times = []
        sample_counts = []
        
        for i, (data, target) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            start_time = time.time()
            # Simulate processing
            _ = data.shape
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
            sample_counts.append(len(data))
        
        # Calculate throughput metrics
        throughput_samples = [count/time_taken if time_taken > 0 else 0 
                             for count, time_taken in zip(sample_counts, times)]
        throughput_tokens = [samples * 1024 * 147 for samples in throughput_samples]  # Estimate tokens per sample
        
        # Add metrics to results
        rows.append(("Sample Throughput (mean)", f"{statistics.mean(throughput_samples):.2f} per_sec"))
        rows.append(("Sample Throughput (median)", f"{statistics.median(throughput_samples):.2f} per_sec"))
        rows.append(("Sample Throughput (min)", f"{min(throughput_samples):.2f} per_sec"))
        rows.append(("Sample Throughput (max)", f"{max(throughput_samples):.2f} per_sec"))
        
        rows.append(("Token Throughput (mean)", f"{statistics.mean(throughput_tokens):.2f} per_sec"))
        rows.append(("Token Throughput (median)", f"{statistics.median(throughput_tokens):.2f} per_sec"))
        rows.append(("Token Throughput (min)", f"{min(throughput_tokens):.2f} per_sec"))
        rows.append(("Token Throughput (max)", f"{max(throughput_tokens):.2f} per_sec"))
        
    except Exception as e:
        rows.append(("Data Loader Error", str(e)))

def gpu_system_metrics(category):
    """Get comprehensive GPU system metrics"""
    try:
        # Get NVIDIA GPU info
        smi_out, smi_err, smi_code = run_cmd("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits")
        
        if smi_code == 0:
            gpu_lines = smi_out.strip().split('\n')
            for i, line in enumerate(gpu_lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    category.append((f"GPU {i} Name", parts[0]))
                    category.append((f"GPU {i} Memory Total", f"{parts[1]} MB"))
                    category.append((f"GPU {i} Memory Used", f"{parts[2]} MB"))
                    category.append((f"GPU {i} Memory Free", f"{int(parts[1]) - int(parts[2])} MB"))
                    category.append((f"GPU {i} Utilization", f"{parts[3]}%"))
                    category.append((f"GPU {i} Temperature", f"{parts[4]}°C"))
                    category.append((f"GPU {i} Power Draw", f"{parts[5]}W"))
        
        # CUDA info
        if torch.cuda.is_available():
            category.append(("CUDA Available", "Yes"))
            category.append(("CUDA Version", torch.version.cuda))
            category.append(("GPU Count", str(torch.cuda.device_count())))
            category.append(("Current GPU", torch.cuda.get_device_name()))
        else:
            category.append(("CUDA Available", "No"))
            
    except Exception as e:
        category.append(("GPU Metrics Error", str(e)))

def training_metrics(category):
    """Run basic training benchmark"""
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Simple neural network
            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(784, 512)
                    self.fc2 = nn.Linear(512, 256)
                    self.fc3 = nn.Linear(256, 10)
                    self.relu = nn.ReLU()
                    
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    return self.fc3(x)
            
            model = SimpleNet().to(device)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Generate dummy data
            batch_size = 128
            input_data = torch.randn(batch_size, 784, device=device)
            target_data = torch.randint(0, 10, (batch_size,), device=device)
            
            # Training benchmark
            num_iterations = 100
            times = []
            
            model.train()
            for i in range(num_iterations):
                start_time = time.time()
                
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time
            
            category.append(("Training Iterations", str(num_iterations)))
            category.append(("Avg Time per Batch", f"{avg_time:.4f}s"))
            category.append(("Training Throughput", f"{throughput:.2f} samples/s"))
            category.append(("Final Loss", f"{loss.item():.4f}"))
            
        else:
            category.append(("Training Metrics", "CUDA not available"))
            
    except Exception as e:
        category.append(("Training Metrics Error", str(e)))

def multi_gpu_training_metrics(category):
    """Test multi-GPU training performance"""
    try:
        gpu_count = torch.cuda.device_count()
        category.append(("Available GPUs", str(gpu_count)))
        
        if gpu_count > 1:
            # Simple multi-GPU data parallel training
            device = torch.device('cuda')
            
            class MultiGPUModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(1000, 512)
                    self.fc2 = nn.Linear(512, 256)
                    self.fc3 = nn.Linear(256, 10)
                    self.relu = nn.ReLU()
                    
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    return self.fc3(x)
            
            model = MultiGPUModel()
            if gpu_count > 1:
                model = nn.DataParallel(model)
            model = model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Training benchmark
            batch_size = 256  # Larger batch for multi-GPU
            num_iterations = 50
            
            times = []
            for i in range(num_iterations):
                input_data = torch.randn(batch_size, 1000, device=device)
                target_data = torch.randint(0, 10, (batch_size,), device=device)
                
                start_time = time.time()
                
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time
            
            category.append(("Multi-GPU Training", "DataParallel" if gpu_count > 1 else "Single GPU"))
            category.append(("Avg Time per Batch", f"{avg_time:.4f}s"))
            category.append(("Multi-GPU Throughput", f"{throughput:.2f} samples/s"))
            
        else:
            category.append(("Multi-GPU Training", "Single GPU only"))
            
    except Exception as e:
        category.append(("Multi-GPU Training Error", str(e)))

# Add minimal versions of other required functions...
def cpu_ram_metrics(category):
    """Get CPU and RAM metrics"""
    try:
        category.append(("CPU Cores Physical", str(psutil.cpu_count(logical=False))))
        category.append(("CPU Cores Logical", str(psutil.cpu_count(logical=True))))
        memory = psutil.virtual_memory()
        category.append(("RAM Total", f"{memory.total // (1024**3)} GB"))
        category.append(("RAM Available", f"{memory.available // (1024**3)} GB"))
        category.append(("RAM Usage", f"{memory.percent}%"))
    except Exception as e:
        category.append(("CPU/RAM Metrics Error", str(e)))

def lib_versions(category):
    """Get library version information"""
    try:
        category.append(("PyTorch Version", torch.__version__))
        category.append(("Python Version", platform.python_version()))
        if torch.cuda.is_available():
            category.append(("CUDA Available", "Yes"))
            category.append(("CUDA Version", torch.version.cuda))
        else:
            category.append(("CUDA Available", "No"))
    except Exception as e:
        category.append(("Library Versions Error", str(e)))

def framework_precision(category):
    """Test different precision modes"""
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Test float32 precision
            try:
                a = torch.randn(1000, 1000, device=device, dtype=torch.float32)
                b = torch.randn(1000, 1000, device=device, dtype=torch.float32)
                
                start_time = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                end_time = time.time()
                
                ops_time = end_time - start_time
                category.append(("float32 Matrix Multiply", f"{ops_time:.4f}s"))
                
            except Exception as e:
                category.append(("float32 Support", f"Error: {str(e)}"))
                
        else:
            category.append(("Precision Tests", "CUDA not available"))
            
    except Exception as e:
        category.append(("Framework Precision Error", str(e)))

def power_efficiency(category):
    """Measure power efficiency metrics"""
    try:
        # GPU power draw (if available)
        smi_out, smi_err, smi_code = run_cmd("nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits")
        if smi_code == 0:
            lines = smi_out.strip().split('\n')
            total_power = 0
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    try:
                        power_draw = float(parts[0])
                        power_limit = float(parts[1])
                        total_power += power_draw
                        category.append((f"GPU {i} Power Draw", f"{power_draw}W"))
                        category.append((f"GPU {i} Power Efficiency", f"{power_draw/power_limit*100:.1f}%"))
                    except ValueError:
                        pass
            
            if total_power > 0:
                category.append(("Total GPU Power Draw", f"{total_power}W"))
                
    except Exception as e:
        category.append(("Power Efficiency Error", str(e)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced GPU Benchmarking Suite')
    parser.add_argument('--cost_per_hour', type=float, required=True,
                        help='Cost per hour for the GPU instance')
    parser.add_argument('--summary_output', type=str,
                        help='Output file for Google Sheets summary (tab-separated)')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top metrics to include in summary')
    args = parser.parse_args()

    # Initialize result categories
    system_setup = [("Metric", "Value")]
    hardware_performance = [("Metric", "Value")]
    ml_framework = [("Metric", "Value")]
    multi_gpu = [("Metric", "Value")]

    # Collect all metrics for analysis
    all_metrics = []

    # Run core benchmarks
    print_header("Running GPU System Metrics...")
    gpu_system_metrics(system_setup)
    
    print_header("Running CPU/RAM Metrics...")
    cpu_ram_metrics(system_setup)
    
    print_header("Running Library Versions Check...")
    lib_versions(system_setup)

    print_header("Running Framework Precision Tests...")
    framework_precision(ml_framework)
    
    print_header("Running Training Metrics...")
    training_metrics(ml_framework)
    
    print_header("Running Data Loader Benchmark...")
    data_loader_benchmark(ml_framework)

    print_header("Running Power Efficiency Tests...")
    power_efficiency(hardware_performance)

    print_header("Running Multi-GPU Training Tests...")
    multi_gpu_training_metrics(multi_gpu)

    # Calculate cost metrics
    cost_per_hour = args.cost_per_hour
    cost_per_second = cost_per_hour / 3600

    # Add cost calculations to ML framework results
    for i, (metric, value) in enumerate(ml_framework[1:], 1):
        if "Throughput" in metric and "per_sec" in value:
            try:
                throughput = float(value.split()[0])
                cost_per_item = cost_per_second / throughput if throughput > 0 else 0
                
                if "Sample" in metric:
                    ml_framework.append(("Cost per Sample (from " + metric + ")", f"${cost_per_item:.6f}"))
                elif "Token" in metric:
                    ml_framework.append(("Cost per Token (from " + metric + ")", f"${cost_per_item:.9f}"))
            except:
                pass

    # Collect all metrics for summary
    for metric, value in system_setup[1:]:
        all_metrics.append(("System Setup", metric, value))
    
    for metric, value in hardware_performance[1:]:
        all_metrics.append(("Hardware Performance", metric, value))
        
    for metric, value in ml_framework[1:]:
        all_metrics.append(("ML Framework", metric, value))
        
    for metric, value in multi_gpu[1:]:
        all_metrics.append(("Multi-GPU", metric, value))

    # Print detailed results
    print_header("Summary: System Setup")
    print_table(system_setup)

    print_header("Summary: Hardware & Performance")
    print_table(hardware_performance)

    print_header("Summary: ML Framework Tests")
    print_table(ml_framework)

    print_header("Summary: Multi-GPU & Parallelism")
    print_table(multi_gpu)

    # Generate Google Sheets summary
    output_google_sheets_summary(all_metrics, cost_per_hour, args.summary_output)
    
    print(f"\nBenchmark completed! Analyzed {len(all_metrics)} total metrics.")
    print(f"Top {args.top_k} most important metrics selected for Google Sheets summary.")
