import subprocess
import torch
import time
import os
import platform
import statistics
import json
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def log(title):
    print(f"\n\033[1m=== {title} ===\033[0m")

def check_system():
    log("System Info")
    print("OS:           ", platform.platform())
    print("CPU:          ", platform.processor())
    ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
    print("RAM (approx): ", f"{ram_gb:.2f} GB")

def check_gpu():
    log("GPU + CUDA")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print("CUDA available: True")
        print("GPU Name:       ", name)
        print("VRAM (total):   ", f"{total_mem:.2f} GB")
        return True
    else:
        print("CUDA available: False")
        return False

def hf_inference_benchmark():
    log("HuggingFace Inference Test")
    model_name = "distilbert-base-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        inputs = tokenizer("GPU benchmark test sentence.", return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        duration = time.time() - t0
        print(f"Inference time: {duration:.4f} sec")
        return {
            "benchmark": "< 0.3 sec inference",
            "result": f"{duration:.4f} sec",
            "pass": duration < 0.3,
            "definition": "Load DistilBERT and run one inference"
        }
    except Exception as e:
        print("✖ HuggingFace inference failed:", e)
        return {
            "benchmark": "< 0.3 sec inference",
            "result": "Error",
            "pass": False,
            "definition": "Load DistilBERT and run one inference"
        }

def check_kubernetes():
    log("Kubernetes Support")
    res = subprocess.run("kubectl version --client", shell=True,
                         capture_output=True, text=True)
    if res.returncode == 0 and "Client Version" in res.stdout:
        version = next(line for line in res.stdout.splitlines() if "Client Version" in line)
        print("✔", version.strip())
        return True
    else:
        print("✖ kubectl CLI not found or broken")
        return False

def check_uptime():
    log("Uptime Check")
    try:
        up_s = float(open("/proc/uptime").read().split()[0])
        hrs = up_s / 3600
        print(f"System uptime: {hrs:.1f} hours")
        return hrs >= 48
    except Exception as e:
        print("✖ Error reading uptime:", e)
        return False

def check_network():
    log("Network Ping Test")
    subprocess.run("ping -c4 google.com", shell=True)

def check_speedtest():
    log("Speedtest (python3 -m speedtest --simple)")
    try:
        res = subprocess.run("python3 -m speedtest --simple", shell=True,
                             capture_output=True, text=True, check=True)
        out = res.stdout.strip()
        print(out)
        download = next((float(l.split()[1]) for l in out.splitlines() if l.startswith("Download:")), None)
        return {
            "benchmark": "> 100 Mbps",
            "result": f"{download} Mbps" if download else "Unknown",
            "pass": bool(download and download > 100),
            "definition": "Download throughput via speedtest.net CLI"
        }
    except Exception as e:
        print("✖ Speedtest failed:", e)
        return {
            "benchmark": "> 100 Mbps",
            "result": "Error",
            "pass": False,
            "definition": "Download throughput via speedtest.net CLI"
        }

def check_disk_io():
    log("Disk Write Speed")
    subprocess.run("dd if=/dev/zero of=benchmark.tmp bs=1G count=1 oflag=direct", shell=True)
    log("Disk Read Speed")
    subprocess.run("dd if=benchmark.tmp of=/dev/null bs=1G count=1 iflag=direct", shell=True)
    os.remove("benchmark.tmp")

def check_us_latency():
    log("US Latency Test")
    try:
        res = subprocess.run("ping -c4 google.com", shell=True,
                             capture_output=True, text=True)
        if res.returncode == 0:
            lat = next(line for line in res.stdout.splitlines() if "rtt" in line)
            avg = lat.split("/")[4]
            print(f"✔ avg latency to google.com: {avg} ms")
            return float(avg) < 100
    except Exception as e:
        print("✖ US Latency check failed:", e)
    return False

def check_nccl():
    log("NCCL Test")
    res = subprocess.run("which nccol_test", shell=True,
                         capture_output=True, text=True)
    if res.returncode == 0:
        print("✔ NCCOL binary found at:", res.stdout.strip())
        return True
    else:
        print("✖ NCCOL binary not found")
        return False

def detect_multi_gpu():
    log("Multi-GPU Detection")
    count = torch.cuda.device_count()
    lines = [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(count)]
    for ln in lines: print(" -", ln)
    return {
        "benchmark": ">= 2 GPUs",
        "result": f"{count} GPU(s): " + "; ".join(lines),
        "pass": count >= 2,
        "definition": "Detect and list all GPUs"
    }

def check_gpu_topology():
    log("GPU Interconnect Topology")
    try:
        topo = subprocess.check_output("nvidia-smi topo --matrix", shell=True, text=True)
        print(topo)
        return {
            "benchmark": "Topology matrix shown",
            "result": "nvidia-smi topo --matrix output",
            "pass": True,
            "definition": "Show NVLink/PCIe interconnects"
        }
    except Exception as e:
        print("✖ Topology check failed:", e)
        return {
            "benchmark": "Topology matrix shown",
            "result": "Error",
            "pass": False,
            "definition": "Show NVLink/PCIe interconnects"
        }

def benchmark_throughput(repeats=5):
    log("GPU Throughput Benchmark")
    times = []
    try:
        for i in range(repeats):
            a = torch.rand((10000,10000), device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            _ = a @ a
            torch.cuda.synchronize()
            dt = time.time() - t0
            times.append(dt)
            print(f"Run {i+1}: {dt:.4f} sec")
        avg = statistics.mean(times)
        md = max(times)
        sd = statistics.stdev(times) if repeats>1 else 0.0
        print(f"→ avg={avg:.4f}s, max={md:.4f}s, stddev={sd:.4f}s")
        return {
            "benchmark": "≤ 3.0 sec (4090 expected)",
            "result": f"avg={avg:.4f}s,max={md:.4f}s,stddev={sd:.4f}s",
            "pass": md <= 3.0,
            "definition": "5× 10k×10k matmul consistency"
        }
    except Exception as e:
        print("✖ Throughput bench failed:", e)
        return {
            "benchmark": "≤ 3.0 sec (4090 expected)",
            "result": "Error",
            "pass": False,
            "definition": "5× 10k×10k matmul consistency"
        }

def print_detailed_scorecard(results):
    log("VALIDATION SCORECARD")
    headers = ["Test","Benchmark","Result","Status","Definition"]
    rows = []
    for test, data in results.items():
        status = "✔" if data["pass"] else "✖"
        rows.append([
            test,
            data.get("benchmark",""),
            data.get("result",""),
            status,
            data.get("definition","")
        ])
    # compute column widths
    widths = [max(len(str(col)) for col in ([headers] + rows)[i]) for i in range(len(headers))]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-"*w for w in widths))
    for row in rows:
        print(fmt.format(*row))

if __name__ == "__main__":
    # Build results dict
    results = {
        "GPU + CUDA": {
            "benchmark": "CUDA available + GPU detected",
            "result": f"CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}",
            "pass": torch.cuda.is_available(),
            "definition": "Confirms GPU visibility and CUDA"
        },
        "HuggingFace Inference": hf_inference_benchmark(),
        "Uptime": {
            "benchmark": "≥ 48 hours",
            "result": f"{round(float(open('/proc/uptime').read().split()[0]) / 3600,1)} hrs",
            "pass": check_uptime(),
            "definition": "Machine uptime ≥ 48 hrs"
        },
        "Kubernetes CLI": {
            "benchmark": "kubectl installed",
            "result": "",
            "pass": check_kubernetes(),
            "definition": "kubectl version check"
        },
        "US Latency": {
            "benchmark": "< 100 ms",
            "result": "",
            "pass": check_us_latency(),
            "definition": "Ping google.com"
        },
        "NCCOL": {
            "benchmark": "nccol_test binary found",
            "result": "",
            "pass": check_nccl(),
            "definition": "which nccol_test"
        },
        "Download Speed": check_speedtest(),
        "GPU Throughput": benchmark_throughput(),
        "Multi-GPU Detection": detect_multi_gpu(),
        "GPU Topology": check_gpu_topology(),
    }

    # Intermediate prints
    check_system()
    check_gpu()
    check_network()
    check_disk_io()

    # Final summary
    print_detailed_scorecard(results)

    # Save results
    with open("gpu_benchmark_results.json","w") as f:
        json.dump(results,f,indent=2)
    with open("gpu_benchmark_results.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test","Benchmark","Result","Status","Definition"])
        for t,d in results.items():
            writer.writerow([
                t,
                d.get("benchmark",""),
                d.get("result",""),
                "PASS" if d["pass"] else "FAIL",
                d.get("definition","")
            ])
    print("\n✔ Results saved to gpu_benchmark_results.json and .csv")
