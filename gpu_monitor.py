"""
GPU Monitoring Script for RTX 4060
Run this in a separate terminal to monitor GPU usage during training
"""
import subprocess
import time
import os

def monitor_gpu():
    print("🔍 GPU Monitoring Started")
    print("Run this while training to see real-time GPU usage")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    try:
        while True:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("🚀 RTX 4060 Real-time Monitoring")
            print("=" * 50)
            
            # Run nvidia-smi command
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 7:
                    name = gpu_info[0]
                    gpu_util = gpu_info[1]
                    mem_util = gpu_info[2]
                    mem_used = gpu_info[3]
                    mem_total = gpu_info[4]
                    temp = gpu_info[5]
                    power = gpu_info[6]
                    
                    print(f"GPU: {name}")
                    print(f"GPU Utilization: {gpu_util}%")
                    print(f"Memory Utilization: {mem_util}%")
                    print(f"Memory Used: {mem_used}MB / {mem_total}MB")
                    print(f"Temperature: {temp}°C")
                    print(f"Power Draw: {power}W")
                    
                    # Status indicators
                    gpu_util_int = int(gpu_util)
                    if gpu_util_int >= 95:
                        print("🟢 EXCELLENT: GPU at maximum utilization!")
                    elif gpu_util_int >= 80:
                        print("🟡 GOOD: High GPU utilization")
                    elif gpu_util_int >= 50:
                        print("🟠 MODERATE: Medium GPU utilization")
                    else:
                        print("🔴 LOW: GPU underutilized")
                        
            else:
                print("❌ Could not get GPU information")
                print("Make sure NVIDIA drivers are installed")
            
            print("-" * 50)
            print("Refreshing in 2 seconds... (Ctrl+C to stop)")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n👋 GPU monitoring stopped")

if __name__ == "__main__":
    monitor_gpu()