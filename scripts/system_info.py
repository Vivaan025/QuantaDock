#!/usr/bin/env python3
"""
System Information and Hardware Specifications Checker
Displays detailed CPU, GPU, and memory specifications for ML optimization
"""

import torch
import psutil
import platform
import subprocess
import json
from pathlib import Path

def get_cpu_info():
    """Get detailed CPU information"""
    cpu_info = {
        'processor': platform.processor(),
        'architecture': platform.architecture()[0],
        'cores_physical': psutil.cpu_count(logical=False),
        'cores_logical': psutil.cpu_count(logical=True),
        'frequency_max': psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A',
        'frequency_current': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A',
    }
    
    # Try to get more detailed CPU info from /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            for line in cpuinfo.split('\n'):
                if 'model name' in line:
                    cpu_info['model'] = line.split(':')[1].strip()
                    break
    except:
        cpu_info['model'] = 'Unknown'
    
    return cpu_info

def get_memory_info():
    """Get detailed memory information"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    memory_info = {
        'total_gb': round(mem.total / (1024**3), 2),
        'available_gb': round(mem.available / (1024**3), 2),
        'used_gb': round(mem.used / (1024**3), 2),
        'percentage_used': mem.percent,
        'swap_total_gb': round(swap.total / (1024**3), 2),
        'swap_used_gb': round(swap.used / (1024**3), 2),
    }
    
    # Try to get memory bandwidth info
    try:
        result = subprocess.run(['dmidecode', '-t', 'memory'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Parse DMI output for memory speed
            for line in result.stdout.split('\n'):
                if 'Speed:' in line and 'MHz' in line:
                    memory_info['speed_mhz'] = line.split(':')[1].strip()
                    break
    except:
        memory_info['speed_mhz'] = 'Unknown'
    
    return memory_info

def get_gpu_info():
    """Get detailed GPU information"""
    gpu_info = []
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.device(i)
            props = torch.cuda.get_device_properties(i)
            
            # Get memory info
            mem_info = torch.cuda.mem_get_info(i)
            
            gpu_data = {
                'device_id': i,
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'total_memory_gb': round(props.total_memory / (1024**3), 2),
                'available_memory_gb': round(mem_info[0] / (1024**3), 2),
                'used_memory_gb': round((props.total_memory - mem_info[0]) / (1024**3), 2),
                'multiprocessor_count': props.multiprocessor_count,
                'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
                'max_threads_per_block': props.max_threads_per_block,
                'warp_size': props.warp_size,
                'memory_bandwidth_gb_s': 'Unknown',  # Will try to get from nvidia-ml-py
            }
            
            # Try to get more detailed GPU info using nvidia-ml-py
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory bandwidth
                mem_info_detailed = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_data['memory_bandwidth_gb_s'] = 'Calculated from specs'
                
                # Clock speeds
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                gpu_data['gpu_clock_mhz'] = gpu_clock
                gpu_data['memory_clock_mhz'] = mem_clock
                
                # Power info
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                gpu_data['power_limit_watts'] = power_limit[1] / 1000  # Convert mW to W
                
            except ImportError:
                print("pynvml not available, install with: pip install nvidia-ml-py")
            except Exception as e:
                print(f"Error getting detailed GPU info: {e}")
            
            gpu_info.append(gpu_data)
    
    return gpu_info

def estimate_model_performance():
    """Estimate performance characteristics for different model types"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device('cuda:0')
    performance_estimates = {}
    
    # Test tensor operations for throughput estimation
    try:
        # Matrix multiplication test (simulates transformer operations)
        size = 4096
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        for _ in range(10):
            torch.matmul(a, b)
        
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(100):
            torch.matmul(a, b)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        
        # Calculate FLOPS (approximate)
        flops = (2 * size**3 * 100) / elapsed_time  # 2*N^3 operations per matmul
        performance_estimates['matmul_gflops'] = round(flops / 1e9, 2)
        
        # Memory bandwidth test
        size_mb = 1024  # 1GB test
        data = torch.randn(size_mb * 1024 * 1024 // 4, device=device)  # 4 bytes per float32
        
        start_time.record()
        for _ in range(10):
            data.copy_(data)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000
        bandwidth = (size_mb * 10) / elapsed_time  # MB/s
        performance_estimates['memory_bandwidth_gb_s'] = round(bandwidth / 1024, 2)
        
    except Exception as e:
        performance_estimates['error'] = str(e)
    
    return performance_estimates

def calculate_quantization_tops():
    """Calculate theoretical TOPS for different quantizations"""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return {}
    
    # Approximate TOPS calculations based on GPU specs
    # These are rough estimates based on compute capability and core count
    tops_estimates = {}
    
    for gpu in gpu_info:
        gpu_name = gpu['name']
        compute_cap = gpu['compute_capability']
        sm_count = gpu['multiprocessor_count']
        
        # Base TOPS estimation (very approximate)
        # Modern GPUs: RTX 4090 ~83 TOPS INT8, RTX 3090 ~35 TOPS INT8
        base_ops_per_sm = {
            '8.6': 256,  # RTX 30 series
            '8.9': 512,  # RTX 40 series
            '7.5': 128,  # GTX 16 series
        }
        
        ops_per_sm = base_ops_per_sm.get(compute_cap, 64)
        base_freq_ghz = 1.5  # Approximate base frequency
        
        # Calculate theoretical TOPS for different precisions
        fp32_tops = (sm_count * ops_per_sm * base_freq_ghz) / 1000
        fp16_tops = fp32_tops * 2
        int8_tops = fp32_tops * 4
        int4_tops = fp32_tops * 8
        
        tops_estimates[gpu_name] = {
            'fp32_tops': round(fp32_tops, 2),
            'fp16_tops': round(fp16_tops, 2),
            'int8_tops': round(int8_tops, 2),
            'int4_tops': round(int4_tops, 2),
        }
    
    return tops_estimates

def main():
    """Main function to display all system information"""
    print("=" * 80)
    print("ML OPTIMIZATION SYSTEM SPECIFICATIONS")
    print("=" * 80)
    
    # CPU Information
    print("\n CPU SPECIFICATIONS")
    print("-" * 40)
    cpu_info = get_cpu_info()
    for key, value in cpu_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Memory Information
    print("\n MEMORY SPECIFICATIONS")
    print("-" * 40)
    memory_info = get_memory_info()
    for key, value in memory_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # GPU Information
    print("\n GPU SPECIFICATIONS")
    print("-" * 40)
    gpu_info = get_gpu_info()
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            print(f"\nGPU {i}:")
            for key, value in gpu.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("No CUDA-capable GPUs found")
    
    # Performance Estimates
    print("\n PERFORMANCE ESTIMATES")
    print("-" * 40)
    performance = estimate_model_performance()
    for key, value in performance.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Quantization TOPS
    print("\n QUANTIZATION PERFORMANCE (TOPS)")
    print("-" * 40)
    tops = calculate_quantization_tops()
    for gpu_name, specs in tops.items():
        print(f"\n{gpu_name}:")
        for precision, tops_value in specs.items():
            print(f"  {precision.upper()}: {tops_value} TOPS")
    
    # Save specs to file
    all_specs = {
        'cpu': cpu_info,
        'memory': memory_info,
        'gpu': gpu_info,
        'performance': performance,
        'quantization_tops': tops
    }
    
    with open('/workspace/system_specs.json', 'w') as f:
        json.dump(all_specs, f, indent=2)
    
    print(f"\n Specifications saved to: /workspace/system_specs.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
