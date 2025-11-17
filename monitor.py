"""
Real-time performance monitor cho pipeline.
Chạy trong terminal riêng để theo dõi CPU, RAM, GPU usage.
"""
import time
import sys
import os

try:
    import psutil
except ImportError:
    print("❌ Thiếu module psutil")
    print("Cài đặt: pip install psutil")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_gpu_info():
    """Lấy thông tin GPU nếu có"""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None
    
    gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
    gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return {
        'allocated': gpu_mem_allocated,
        'reserved': gpu_mem_reserved,
        'total': gpu_mem_total,
        'utilization': (gpu_mem_allocated / gpu_mem_total) * 100
    }

def clear_screen():
    """Clear terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("="*70)
    print("🔍 PERFORMANCE MONITOR - Audio Transcription Pipeline")
    print("="*70)
    print("\nĐang khởi động...")
    time.sleep(1)
    
    interval = 1  # Update every second
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("="*70)
            print("🔍 PERFORMANCE MONITOR")
            print("="*70)
            print(f"⏰ {time.strftime('%H:%M:%S')}")
            print()
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            print("🖥️  CPU:")
            print(f"   Usage:     {cpu_percent:5.1f}%")
            print(f"   Cores:     {cpu_count}")
            if cpu_freq:
                print(f"   Frequency: {cpu_freq.current:.0f} MHz")
            
            # CPU bar
            cpu_bar_length = int(cpu_percent / 2)
            cpu_bar = "█" * cpu_bar_length + "░" * (50 - cpu_bar_length)
            print(f"   [{cpu_bar}] {cpu_percent:.1f}%")
            print()
            
            # RAM
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            
            print("💾 RAM:")
            print(f"   Used:  {ram_used_gb:5.1f} GB / {ram_total_gb:.1f} GB")
            print(f"   Free:  {ram.available / (1024**3):5.1f} GB")
            print(f"   Usage: {ram.percent:5.1f}%")
            
            # RAM bar
            ram_bar_length = int(ram.percent / 2)
            ram_bar = "█" * ram_bar_length + "░" * (50 - ram_bar_length)
            print(f"   [{ram_bar}] {ram.percent:.1f}%")
            print()
            
            # GPU
            gpu_info = get_gpu_info()
            if gpu_info:
                print("🎮 GPU (CUDA):")
                print(f"   Allocated: {gpu_info['allocated']:5.2f} GB")
                print(f"   Reserved:  {gpu_info['reserved']:5.2f} GB")
                print(f"   Total:     {gpu_info['total']:5.2f} GB")
                print(f"   Usage:     {gpu_info['utilization']:5.1f}%")
                
                # GPU bar
                gpu_bar_length = int(gpu_info['utilization'] / 2)
                gpu_bar = "█" * gpu_bar_length + "░" * (50 - gpu_bar_length)
                print(f"   [{gpu_bar}] {gpu_info['utilization']:.1f}%")
            else:
                print("🎮 GPU: Không có hoặc không sử dụng")
            print()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                read_mb = disk_io.read_bytes / (1024**2)
                write_mb = disk_io.write_bytes / (1024**2)
                print("💿 Disk I/O:")
                print(f"   Read:  {read_mb:10.1f} MB")
                print(f"   Write: {write_mb:10.1f} MB")
                print()
            
            # Process info
            python_procs = [p for p in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']) 
                           if 'python' in p.info['name'].lower()]
            
            if python_procs:
                print("🐍 Python Processes:")
                for proc in python_procs[:5]:  # Top 5
                    print(f"   PID {proc.pid:6d}: CPU {proc.info['cpu_percent']:5.1f}% | RAM {proc.info['memory_percent']:5.1f}%")
                print()
            
            print("="*70)
            print("💡 Press Ctrl+C to exit")
            print("="*70)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitor stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
