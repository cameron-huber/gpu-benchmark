#!/usr/bin/env python3
"""
GPU Benchmarking Suite
Comprehensive GPU and system performance benchmarking tool
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

def container_checks(rows):
    """Check container and orchestration compatibility"""
    print_header("Container & Orchestration Compatibility")
    
    # Check Docker
    docker_out, docker_err, docker_code = run_cmd("docker --version")
    docker_installed = docker_code == 0
    docker_version = docker_out.split()[2].rstrip(',') if docker_installed else "N/A"
    
    # Check Kubernetes CLI
    kubectl_out, kubectl_err, kubectl_code = run_cmd("kubectl version --client")
    kubectl_installed = kubectl_code == 0
    kubectl_version = "Available" if kubectl_installed else "N/A"
    
    # Check GPU visibility in containers
    gpu_in_container = False
    if docker_installed:
        gpu_out, gpu_err, gpu_code = run_cmd("docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi")
        gpu_in_container = gpu_code == 0
    
    rows.append(("Docker Installed", str(docker_installed)))
    rows.append(("Docker Version", docker_version))
    rows.append(("Kubernetes CLI Installed", str(kubectl_installed)))
    rows.append(("Kubernetes CLI Version", kubectl_version))
    rows.append(("GPU Visibility in Container", str(gpu_in_container)))

def compare_envs(images):
    """Compare performance across different container environments"""
    results = {}
    for image in images:
        try:
            cmd = f"docker run --rm --gpus all {image} python -c 'import torch; print(torch.cuda.is_available())'"
            out, err, code = run_cmd(cmd)
            results[image] = "PASS" if code == 0 and "True" in out else "FAIL"
        except:
            results[image] = "FAIL"
    return results

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

def cpu_ram_metrics(category):
    """Get CPU and RAM metrics"""
    try:
        # CPU info
        category.append(("CPU Model", platform.processor() or "Unknown"))
        category.append(("CPU Cores Physical", str(psutil.cpu_count(logical=False))))
        category.append(("CPU Cores Logical", str(psutil.cpu_count(logical=True))))
        category.append(("CPU Frequency", f"{psutil.cpu_freq().max:.0f} MHz"))
        
        # CPU usage over 1 second
        cpu_percent = psutil.cpu_percent(interval=1)
        category.append(("CPU Usage", f"{cpu_percent}%"))
        
        # Memory info
        memory = psutil.virtual_memory()
        category.append(("RAM Total", f"{memory.total // (1024**3)} GB"))
        category.append(("RAM Available", f"{memory.available // (1024**3)} GB"))
        category.append(("RAM Used", f"{memory.used // (1024**3)} GB"))
        category.append(("RAM Usage", f"{memory.percent}%"))
        
        # Load average (Linux/Mac)
        try:
            load_avg = os.getloadavg()
            category.append(("Load Average (1m)", f"{load_avg[0]:.2f}"))
            category.append(("Load Average (5m)", f"{load_avg[1]:.2f}"))
            category.append(("Load Average (15m)", f"{load_avg[2]:.2f}"))
        except:
            pass
            
    except Exception as e:
        category.append(("CPU/RAM Metrics Error", str(e)))

def lib_versions(category):
    """Get library version information"""
    try:
        # PyTorch and related
        category.append(("PyTorch Version", torch.__version__))
        category.append(("TorchVision Version", torchvision.__version__))
        
        # Python and system
        category.append(("Python Version", platform.python_version()))
        category.append(("Platform", platform.platform()))
        
        # Try to get other common ML libraries
        try:
            import transformers
            category.append(("Transformers Version", transformers.__version__))
        except:
            category.append(("Transformers Version", "Not installed"))
            
        try:
            import numpy
            category.append(("NumPy Version", numpy.__version__))
        except:
            category.append(("NumPy Version", "Not installed"))
            
        # NVIDIA driver version
        driver_out, driver_err, driver_code = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        if driver_code == 0:
            category.append(("NVIDIA Driver", driver_out.strip()))
            
    except Exception as e:
        category.append(("Library Versions Error", str(e)))

def disk_io(category):
    """Test disk I/O performance"""
    try:
        # Get disk usage
        disk_usage = psutil.disk_usage('/')
        category.append(("Disk Total", f"{disk_usage.total // (1024**3)} GB"))
        category.append(("Disk Used", f"{disk_usage.used // (1024**3)} GB"))
        category.append(("Disk Free", f"{disk_usage.free // (1024**3)} GB"))
        category.append(("Disk Usage", f"{disk_usage.used/disk_usage.total*100:.1f}%"))
        
        # Simple write/read test
        test_file = "/tmp/gpu_benchmark_disktest"
        test_data = b"0" * (1024 * 1024)  # 1MB
        
        # Write test
        start_time = time.time()
        with open(test_file, 'wb') as f:
            for _ in range(100):  # Write 100MB
                f.write(test_data)
        write_time = time.time() - start_time
        write_speed = 100 / write_time
        
        # Read test
        start_time = time.time()
        with open(test_file, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        read_time = time.time() - start_time
        read_speed = 100 / read_time
        
        category.append(("Disk Write Speed", f"{write_speed:.1f} MB/s"))
        category.append(("Disk Read Speed", f"{read_speed:.1f} MB/s"))
        
        # Cleanup
        os.remove(test_file)
        
    except Exception as e:
        category.append(("Disk I/O Error", str(e)))

def network_ping(category):
    """Test network connectivity and latency"""
    try:
        # Test ping to common hosts
        hosts = ["8.8.8.8", "1.1.1.1", "google.com"]
        
        for host in hosts:
            ping_out, ping_err, ping_code = run_cmd(f"ping -c 3 {host}")
            if ping_code == 0:
                # Extract average ping time
                lines = ping_out.split('\n')
                for line in lines:
                    if 'avg' in line and 'ms' in line:
                        # Parse ping statistics line
                        parts = line.split('=')
                        if len(parts) > 1:
                            times = parts[1].split('/')
                            if len(times) >= 2:
                                avg_time = times[1]
                                category.append((f"Ping {host}", f"{avg_time} ms"))
                                break
                else:
                    category.append((f"Ping {host}", "Success"))
            else:
                category.append((f"Ping {host}", "Failed"))
                
        # Network interface info
        interfaces = psutil.net_if_addrs()
        active_interfaces = [name for name, addrs in interfaces.items() 
                           if any(addr.family == socket.AF_INET for addr in addrs)]
        category.append(("Network Interfaces", ", ".join(active_interfaces)))
        
    except Exception as e:
        category.append(("Network Test Error", str(e)))

def uptime(category):
    """Get system uptime information"""
    try:
        # System boot time
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        current_time = datetime.now()
        uptime_delta = current_time - boot_time
        
        days = uptime_delta.days
        hours, remainder = divmod(uptime_delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        category.append(("System Boot Time", boot_time.strftime("%Y-%m-%d %H:%M:%S")))
        category.append(("System Uptime", f"{days}d {hours}h {minutes}m"))
        
        # Process count
        category.append(("Running Processes", str(len(psutil.pids()))))
        
    except Exception as e:
        category.append(("Uptime Error", str(e)))

def power_efficiency(category):
    """Measure power efficiency metrics"""
    try:
        # GPU power draw (if available)
        smi_out, smi_err, smi_code = run_cmd("nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits")
        if smi_code == 0:
            lines = smi_out.strip().split('\n')
            total_power = 0
            total_limit = 0
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    try:
                        power_draw = float(parts[0])
                        power_limit = float(parts[1])
                        total_power += power_draw
                        total_limit += power_limit
                        category.append((f"GPU {i} Power Draw", f"{power_draw}W"))
                        category.append((f"GPU {i} Power Limit", f"{power_limit}W"))
                        category.append((f"GPU {i} Power Efficiency", f"{power_draw/power_limit*100:.1f}%"))
                    except ValueError:
                        pass
            
            if total_limit > 0:
                category.append(("Total GPU Power Draw", f"{total_power}W"))
                category.append(("Total GPU Power Limit", f"{total_limit}W"))
                category.append(("Overall Power Efficiency", f"{total_power/total_limit*100:.1f}%"))
        
        # System power info (if available)
        try:
            battery = psutil.sensors_battery()
            if battery:
                category.append(("Battery Level", f"{battery.percent}%"))
                category.append(("Power Plugged", str(battery.power_plugged)))
        except:
            pass
            
    except Exception as e:
        category.append(("Power Efficiency Error", str(e)))

def ib_bandwidth(category):
    """Test InfiniBand bandwidth if available"""
    try:
        # Check for InfiniBand devices
        ib_out, ib_err, ib_code = run_cmd("ibstat")
        if ib_code == 0:
            category.append(("InfiniBand Status", "Available"))
            
            # Try to get IB device info
            ibv_out, ibv_err, ibv_code = run_cmd("ibv_devinfo")
            if ibv_code == 0:
                lines = ibv_out.split('\n')
                for line in lines:
                    if 'device:' in line:
                        device = line.split(':')[1].strip()
                        category.append(("InfiniBand Device", device))
                        break
                        
            # Simple bandwidth test (if ibv_bw tools available)
            bw_out, bw_err, bw_code = run_cmd("which ib_send_bw")
            if bw_code == 0:
                category.append(("InfiniBand Bandwidth Tools", "Available"))
            else:
                category.append(("InfiniBand Bandwidth Tools", "Not Available"))
        else:
            category.append(("InfiniBand Status", "Not Available"))
            
    except Exception as e:
        category.append(("InfiniBand Error", str(e)))

def nvlink_bandwidth(category):
    """Test NVLink bandwidth between GPUs"""
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Test P2P access between GPUs
            gpu_count = torch.cuda.device_count()
            category.append(("GPU Count for NVLink", str(gpu_count)))
            
            p2p_matrix = []
            for i in range(gpu_count):
                row = []
                for j in range(gpu_count):
                    if i != j:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        row.append("Yes" if can_access else "No")
                    else:
                        row.append("N/A")
                p2p_matrix.append(row)
                category.append((f"GPU {i} P2P Access", " ".join(row)))
            
            # Simple bandwidth test between GPUs
            if gpu_count >= 2:
                try:
                    # Create tensors on different GPUs
                    device0 = torch.device('cuda:0')
                    device1 = torch.device('cuda:1')
                    
                    # 100MB tensor
                    tensor_size = 25 * 1024 * 1024  # 100MB of float32
                    data = torch.randn(tensor_size, device=device0)
                    
                    # Time transfer
                    torch.cuda.synchronize()
                    start_time = time.time()
                    data_gpu1 = data.to(device1)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    transfer_time = end_time - start_time
                    bandwidth = (tensor_size * 4) / (transfer_time * 1024**3)  # GB/s
                    category.append(("NVLink Bandwidth Test", f"{bandwidth:.2f} GB/s"))
                    
                except Exception as e:
                    category.append(("NVLink Bandwidth Test", f"Error: {str(e)}"))
        else:
            category.append(("NVLink Status", "Single GPU or CUDA not available"))
            
    except Exception as e:
        category.append(("NVLink Error", str(e)))

def framework_precision(category):
    """Test different precision modes"""
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Test different precisions
            precisions = [
                ('float32', torch.float32),
                ('float16', torch.float16),
                ('bfloat16', torch.bfloat16) if hasattr(torch, 'bfloat16') else None
            ]
            
            for name, dtype in precisions:
                if dtype is None:
                    continue
                    
                try:
                    # Simple tensor operations
                    a = torch.randn(1000, 1000, device=device, dtype=dtype)
                    b = torch.randn(1000, 1000, device=device, dtype=dtype)
                    
                    start_time = time.time()
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    ops_time = end_time - start_time
                    category.append((f"{name} Matrix Multiply", f"{ops_time:.4f}s"))
                    
                except Exception as e:
                    category.append((f"{name} Support", f"Error: {str(e)}"))
                    
            # Mixed precision support
            try:
                scaler = torch.cuda.amp.GradScaler()
                category.append(("Mixed Precision Support", "Available"))
            except:
                category.append(("Mixed Precision Support", "Not Available"))
                
        else:
            category.append(("Precision Tests", "CUDA not available"))
            
    except Exception as e:
        category.append(("Framework Precision Error", str(e)))

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

def huggingface_inference(category):
    """Test Hugging Face model inference"""
    try:
        # Use a small model for testing
        model_name = "distilbert-base-uncased"
        
        # Text classification pipeline
        classifier = pipeline("sentiment-analysis", model=model_name, device=0 if torch.cuda.is_available() else -1)
        
        # Test sentences
        test_texts = [
            "This is a great day!",
            "I'm feeling sad today.",
            "The weather is neutral.",
            "Amazing performance on this GPU!",
            "This benchmark is running well."
        ]
        
        # Benchmark inference
        start_time = time.time()
        results = classifier(test_texts)
        end_time = time.time()
        
        inference_time = end_time - start_time
        throughput = len(test_texts) / inference_time
        
        category.append(("HuggingFace Model", model_name))
        category.append(("Inference Time", f"{inference_time:.4f}s"))
        category.append(("Inference Throughput", f"{throughput:.2f} samples/s"))
        category.append(("Sample Prediction", f"{results[0]['label']}: {results[0]['score']:.3f}"))
        
    except Exception as e:
        category.append(("HuggingFace Inference Error", str(e)))

def proxy_model_training(category):
    """Train a proxy model to test training pipeline"""
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Simple CNN for CIFAR-like data
            class ProxyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(64 * 8 * 8, 128)
                    self.fc2 = nn.Linear(128, 10)
                    self.relu = nn.ReLU()
                    
                def forward(self, x):
                    x = self.pool(self.relu(self.conv1(x)))
                    x = self.pool(self.relu(self.conv2(x)))
                    x = x.view(-1, 64 * 8 * 8)
                    x = self.relu(self.fc1(x))
                    return self.fc2(x)
            
            model = ProxyModel().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Generate dummy CIFAR-like data
            batch_size = 64
            num_batches = 50
            
            total_time = 0
            total_samples = 0
            
            model.train()
            for batch_idx in range(num_batches):
                # Generate random data
                input_data = torch.randn(batch_size, 3, 32, 32, device=device)
                target_data = torch.randint(0, 10, (batch_size,), device=device)
                
                start_time = time.time()
                
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                total_time += (end_time - start_time)
                total_samples += batch_size
            
            avg_throughput = total_samples / total_time
            
            category.append(("Proxy Model Type", "CNN"))
            category.append(("Training Batches", str(num_batches)))
            category.append(("Total Training Time", f"{total_time:.2f}s"))
            category.append(("Training Throughput", f"{avg_throughput:.2f} samples/s"))
            category.append(("Final Loss", f"{loss.item():.4f}"))
            
        else:
            category.append(("Proxy Model Training", "CUDA not available"))
            
    except Exception as e:
        category.append(("Proxy Model Training Error", str(e)))

def generative_model_benchmark(category):
    """Benchmark text generation capabilities"""
    try:
        # Use a small generative model
        generator = pipeline("text-generation", model="distilgpt2", device=0 if torch.cuda.is_available() else -1)
        
        prompt = "The future of AI is"
        
        # Benchmark generation
        start_time = time.time()
        generated = generator(prompt, max_length=50, num_return_sequences=1, do_sample=True)
        end_time = time.time()
        
        generation_time = end_time - start_time
        generated_text = generated[0]['generated_text']
        text_length = len(generated_text)
        
        category.append(("Generative Model", "DistilGPT2"))
        category.append(("Generation Time", f"{generation_time:.4f}s"))
        category.append(("Generated Text Length", f"{text_length} chars"))
        category.append(("Generation Speed", f"{text_length/generation_time:.2f} chars/s"))
        category.append(("Sample Output", generated_text[:100] + "..."))
        
    except Exception as e:
        category.append(("Generative Model Error", str(e)))

def determinism_check(category):
    """Check for deterministic behavior"""
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Set deterministic behavior
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            
            # Run same operation twice
            results = []
            for run in range(2):
                torch.manual_seed(42)
                torch.cuda.manual_seed(42)
                
                a = torch.randn(100, 100, device=device)
                b = torch.randn(100, 100, device=device)
                c = torch.matmul(a, b)
                results.append(c.cpu().numpy())
            
            # Check if results are identical
            are_identical = np.allclose(results[0], results[1], rtol=1e-6)
            max_diff = np.max(np.abs(results[0] - results[1]))
            
            category.append(("Deterministic Results", "Yes" if are_identical else "No"))
            category.append(("Max Difference", f"{max_diff:.2e}"))
            
            # Check CUDA deterministic flags
            category.append(("CUDNN Deterministic", str(torch.backends.cudnn.deterministic)))
            category.append(("CUDNN Benchmark", str(torch.backends.cudnn.benchmark)))
            
        else:
            category.append(("Determinism Check", "CUDA not available"))
            
    except Exception as e:
        category.append(("Determinism Check Error", str(e)))

def k8s_tooling(category):
    """Check Kubernetes tooling availability"""
    try:
        # Check kubectl
        kubectl_out, kubectl_err, kubectl_code = run_cmd("kubectl version --client")
        if kubectl_code == 0:
            category.append(("kubectl Available", "Yes"))
            # Extract version info
            lines = kubectl_out.split('\n')
            for line in lines:
                if 'GitVersion' in line:
                    version = line.split('"')[3] if '"' in line else "Unknown"
                    category.append(("kubectl Version", version))
                    break
        else:
            category.append(("kubectl Available", "No"))
        
        # Check for Kubernetes config
        kube_config = os.path.expanduser("~/.kube/config")
        category.append(("Kube Config Exists", str(os.path.exists(kube_config))))
        
        # Check cluster connectivity (if config exists)
        if os.path.exists(kube_config):
            cluster_out, cluster_err, cluster_code = run_cmd("kubectl cluster-info")
            category.append(("Cluster Connectivity", "Yes" if cluster_code == 0 else "No"))
        else:
            category.append(("Cluster Connectivity", "No config"))
            
        # Check for GPU device plugin availability
        gpu_plugin_out, gpu_plugin_err, gpu_plugin_code = run_cmd("kubectl get daemonset -A | grep nvidia-device-plugin")
        category.append(("GPU Device Plugin", "Detected" if gpu_plugin_code == 0 else "Not Found"))
        
    except Exception as e:
        category.append(("K8s Tooling Error", str(e)))

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

def gradient_sync_latency(category):
    """Measure gradient synchronization latency"""
    try:
        gpu_count = torch.cuda.device_count()
        
        if gpu_count > 1:
            # Test gradient synchronization using DataParallel
            device = torch.device('cuda')
            
            model = nn.Linear(1000, 100)
            model_parallel = nn.DataParallel(model).to(device)
            
            # Create dummy data
            batch_size = 128
            input_data = torch.randn(batch_size, 1000, device=device)
            target_data = torch.randn(batch_size, 100, device=device)
            
            criterion = nn.MSELoss()
            
            # Measure sync time
            sync_times = []
            for i in range(10):
                start_time = time.time()
                
                output = model_parallel(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                
                torch.cuda.synchronize()
                end_time = time.time()
                sync_times.append(end_time - start_time)
            
            avg_sync_time = statistics.mean(sync_times)
            category.append(("Gradient Sync Latency", f"{avg_sync_time:.4f}s"))
            category.append(("GPUs Used for Sync", str(gpu_count)))
            
        else:
            category.append(("Gradient Sync Latency", "Single GPU - N/A"))
            
    except Exception as e:
        category.append(("Gradient Sync Error", str(e)))

def cross_gpu_bandwidth(category):
    """Measure bandwidth between GPUs"""
    try:
        gpu_count = torch.cuda.device_count()
        
        if gpu_count > 1:
            # Test data transfer between GPUs
            tensor_size = 10 * 1024 * 1024  # 40MB tensor (float32)
            
            for i in range(min(gpu_count, 2)):  # Test first 2 GPUs
                for j in range(i + 1, min(gpu_count, 2)):
                    device_i = torch.device(f'cuda:{i}')
                    device_j = torch.device(f'cuda:{j}')
                    
                    # Create tensor on GPU i
                    data = torch.randn(tensor_size, device=device_i)
                    
                    # Measure transfer time
                    torch.cuda.synchronize()
                    start_time = time.time()
                    data_transferred = data.to(device_j)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    transfer_time = end_time - start_time
                    bandwidth = (tensor_size * 4) / (transfer_time * 1024**3)  # GB/s
                    
                    category.append((f"GPU{i}->GPU{j} Bandwidth", f"{bandwidth:.2f} GB/s"))
            
        else:
            category.append(("Cross-GPU Bandwidth", "Single GPU - N/A"))
            
    except Exception as e:
        category.append(("Cross-GPU Bandwidth Error", str(e)))

def multi_gpu_inference(category):
    """Test multi-GPU inference performance"""
    try:
        gpu_count = torch.cuda.device_count()
        
        if gpu_count > 1:
            # Simple model for inference
            model = nn.Sequential(
                nn.Linear(1000, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
            
            # Single GPU inference
            single_gpu_model = model.to('cuda:0')
            
            # Multi-GPU inference (DataParallel)
            multi_gpu_model = nn.DataParallel(model).to('cuda')
            
            batch_size = 512
            input_data = torch.randn(batch_size, 1000, device='cuda')
            
            # Benchmark single GPU
            single_times = []
            for _ in range(20):
                start_time = time.time()
                with torch.no_grad():
                    _ = single_gpu_model(input_data)
                torch.cuda.synchronize()
                single_times.append(time.time() - start_time)
            
            # Benchmark multi GPU
            multi_times = []
            for _ in range(20):
                start_time = time.time()
                with torch.no_grad():
                    _ = multi_gpu_model(input_data)
                torch.cuda.synchronize()
                multi_times.append(time.time() - start_time)
            
            single_avg = statistics.mean(single_times)
            multi_avg = statistics.mean(multi_times)
            speedup = single_avg / multi_avg
            
            category.append(("Single GPU Inference", f"{single_avg:.4f}s"))
            category.append(("Multi GPU Inference", f"{multi_avg:.4f}s"))
            category.append(("Multi-GPU Speedup", f"{speedup:.2f}x"))
            
        else:
            category.append(("Multi-GPU Inference", "Single GPU only"))
            
    except Exception as e:
        category.append(("Multi-GPU Inference Error", str(e)))

def model_parallelism_scaling(category):
    """Test model parallelism scaling"""
    try:
        gpu_count = torch.cuda.device_count()
        
        if gpu_count > 1:
            # Simple pipeline parallel model
            class PipelineModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer1 = nn.Sequential(
                        nn.Linear(1000, 512),
                        nn.ReLU()
                    ).to('cuda:0')
                    
                    self.layer2 = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10)
                    ).to('cuda:1')
                    
                def forward(self, x):
                    x = self.layer1(x.to('cuda:0'))
                    x = self.layer2(x.to('cuda:1'))
                    return x
            
            model = PipelineModel()
            
            # Test pipeline performance
            batch_size = 128
            num_batches = 20
            
            times = []
            for _ in range(num_batches):
                input_data = torch.randn(batch_size, 1000)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_data)
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time
            
            category.append(("Model Parallelism", "Pipeline across 2 GPUs"))
            category.append(("Pipeline Inference Time", f"{avg_time:.4f}s"))
            category.append(("Pipeline Throughput", f"{throughput:.2f} samples/s"))
            
        else:
            category.append(("Model Parallelism", "Single GPU - N/A"))
            
    except Exception as e:
        category.append(("Model Parallelism Error", str(e)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU Benchmarking Suite')
    parser.add_argument('--cost_per_hour', type=float, required=True,
                        help='Cost per hour for the GPU instance')
    parser.add_argument('--env_matrix', action='store_true',
                        help='Run environment matrix comparison')
    args = parser.parse_args()

    # Initialize result categories
    system_setup = [("Metric", "Value")]
    hardware_performance = [("Metric", "Value")]
    ml_framework = [("Metric", "Value")]
    multi_gpu = [("Metric", "Value")]

    # Run all benchmarks
    gpu_system_metrics(system_setup)
    cpu_ram_metrics(system_setup)
    lib_versions(system_setup)

    disk_io(hardware_performance)
    network_ping(hardware_performance)
    uptime(hardware_performance)
    power_efficiency(hardware_performance)
    ib_bandwidth(hardware_performance)
    nvlink_bandwidth(hardware_performance)

    framework_precision(ml_framework)
    training_metrics(ml_framework)
    huggingface_inference(ml_framework)
    proxy_model_training(ml_framework)
    generative_model_benchmark(ml_framework)
    data_loader_benchmark(ml_framework)
    determinism_check(ml_framework)

    k8s_tooling(multi_gpu)
    multi_gpu_training_metrics(multi_gpu)
    gradient_sync_latency(multi_gpu)
    cross_gpu_bandwidth(multi_gpu)
    multi_gpu_inference(multi_gpu)
    model_parallelism_scaling(multi_gpu)
    container_checks(multi_gpu)

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

    # Print results
    print_header("Summary: System Setup")
    print_table(system_setup)

    print_header("Summary: Hardware & Performance")
    print_table(hardware_performance)

    print_header("Summary: ML Framework Tests")
    print_table(ml_framework)

    print_header("Summary: Multi-GPU & Parallelism\nContainer & Orchestration Compatibility")
    print_table(multi_gpu)

    # Environment matrix testing
    if args.env_matrix:
        print_header("Environment Matrix Comparison")
        test_images = [
            'pytorch/pytorch:latest',
            'tensorflow/tensorflow:latest-gpu',
            'nvidia/pytorch:latest'
        ]
        
        results = compare_envs(test_images)
        env_results = [("Container Image", "Status")]
        for image, status in results.items():
            env_results.append((image, status))
        print_table(env_results)
