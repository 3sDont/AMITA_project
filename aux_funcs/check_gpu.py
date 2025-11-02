"""
Script kiểm tra GPU và CUDA cho PyTorch
"""
import torch
import sys


def check_gpu_detailed():
    """Kiểm tra chi tiết GPU và CUDA"""
    print("=" * 80)
    print("🔍 KIỂM TRA GPU VÀ CUDA")
    print("=" * 80)
    
    # PyTorch version
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    print(f"📦 CUDA Version (PyTorch compiled with): {torch.version.cuda}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n🎮 CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if not cuda_available:
        print("\n❌ CUDA KHÔNG KHẢ DỤNG!")
        print("\n💡 Nguyên nhân có thể:")
        print("   1. Chưa cài NVIDIA GPU driver")
        print("   2. Chưa cài CUDA Toolkit")
        print("   3. PyTorch được cài phiên bản CPU-only")
        print("   4. GPU không hỗ trợ CUDA")
        print("\n📝 Hướng dẫn:")
        print("   • Kiểm tra GPU: Mở Task Manager → Performance → GPU")
        print("   • Cài PyTorch GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # GPU details
    print(f"\n🎮 GPU Information:")
    num_gpus = torch.cuda.device_count()
    print(f"   • Số lượng GPU: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"\n   GPU #{i}:")
        print(f"      • Tên: {torch.cuda.get_device_name(i)}")
        print(f"      • Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"      • Tổng VRAM: {total_memory:.2f} GB")
        
        if torch.cuda.memory_allocated(i) > 0 or torch.cuda.memory_reserved(i) > 0:
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"      • VRAM đang dùng: {allocated:.2f} GB")
            print(f"      • VRAM đã reserve: {reserved:.2f} GB")
            print(f"      • VRAM còn trống: {total_memory - reserved:.2f} GB")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n🎯 Current Device: GPU #{current_device} ({torch.cuda.get_device_name(current_device)})")
    
    # cuDNN
    print(f"\n🔧 cuDNN:")
    print(f"   • Enabled: {'✅ YES' if torch.backends.cudnn.enabled else '❌ NO'}")
    print(f"   • Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    return True


def test_gpu_computation():
    """Test tính toán trên GPU"""
    print("\n" + "=" * 80)
    print("🧪 TEST TÍNH TOÁN GPU")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("\n❌ Không thể test - CUDA không khả dụng")
        return
    
    try:
        import time
        
        # Test tensor trên GPU
        print("\n🔄 Tạo tensor trên GPU...")
        device = torch.device("cuda")
        
        # Small test
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"✅ Tính toán ma trận 1000x1000 trên GPU: {gpu_time*1000:.2f}ms")
        
        # Compare with CPU
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        print(f"🖥️  Tính toán ma trận 1000x1000 trên CPU: {cpu_time*1000:.2f}ms")
        print(f"⚡ Tốc độ: GPU nhanh hơn {cpu_time/gpu_time:.2f}x")
        
        # Memory test
        print(f"\n💾 VRAM sau test:")
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"   • Allocated: {allocated:.2f} MB")
        print(f"   • Reserved: {reserved:.2f} MB")
        
        # Clear cache
        del x, y, z, x_cpu, y_cpu, z_cpu
        torch.cuda.empty_cache()
        print(f"   • Sau khi clear cache: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi test GPU: {e}")


def check_whisper_gpu():
    """Kiểm tra Whisper có thể dùng GPU không"""
    print("\n" + "=" * 80)
    print("🎤 KIỂM TRA WHISPER GPU SUPPORT")
    print("=" * 80)
    
    try:
        import whisper
        print(f"\n✅ Whisper version: {whisper.__version__ if hasattr(whisper, '__version__') else 'unknown'}")
        print(f"💡 Whisper tự động dùng GPU nếu torch.cuda.is_available() = True")
        
        if torch.cuda.is_available():
            print(f"✅ Whisper SẼ DÙNG GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️  Whisper sẽ dùng CPU (chậm hơn)")
            
    except ImportError:
        print("\n⚠️  Chưa cài Whisper")


def check_pyannote_gpu():
    """Kiểm tra Pyannote có thể dùng GPU không"""
    print("\n" + "=" * 80)
    print("👥 KIỂM TRA PYANNOTE GPU SUPPORT")
    print("=" * 80)
    
    try:
        import pyannote.audio
        print(f"\n✅ Pyannote.audio version: {pyannote.audio.__version__}")
        print(f"💡 Pyannote tự động dùng GPU nếu torch.cuda.is_available() = True")
        
        if torch.cuda.is_available():
            print(f"✅ Pyannote SẼ DÙNG GPU: {torch.cuda.get_device_name(0)}")
            print(f"\n🔧 Để force sử dụng GPU trong code:")
            print(f'   pipeline = Pipeline.from_pretrained(...)')
            print(f'   pipeline.to(torch.device("cuda"))')
        else:
            print(f"⚠️  Pyannote sẽ dùng CPU (chậm hơn)")
            
    except ImportError:
        print("\n⚠️  Chưa cài Pyannote")


def get_nvidia_smi_info():
    """Lấy thông tin từ nvidia-smi"""
    print("\n" + "=" * 80)
    print("📊 NVIDIA-SMI (Driver Info)")
    print("=" * 80)
    
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            print("\n" + result.stdout)
        else:
            print("\n❌ Không thể chạy nvidia-smi")
            print("💡 Có thể chưa cài NVIDIA driver")
    except FileNotFoundError:
        print("\n❌ Không tìm thấy nvidia-smi")
        print("💡 Vui lòng cài NVIDIA GPU driver từ: https://www.nvidia.com/drivers")
    except subprocess.TimeoutExpired:
        print("\n⚠️  nvidia-smi timeout")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")


def show_recommendations():
    """Hiển thị khuyến nghị"""
    print("\n" + "=" * 80)
    print("💡 KHUYẾN NGHỊ")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("\n❌ GPU CHƯA SẴN SÀNG - Cần làm:")
        print("\n1️⃣  Cài NVIDIA Driver:")
        print("   • Download từ: https://www.nvidia.com/drivers")
        print("   • Hoặc GeForce Experience (tự động)")
        
        print("\n2️⃣  Cài lại PyTorch với CUDA:")
        print("   • Gỡ phiên bản hiện tại:")
        print("     pip uninstall torch torchvision torchaudio")
        print("   • Cài phiên bản GPU (CUDA 12.1):")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   • Hoặc CUDA 11.8:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n3️⃣  Verify lại:")
        print("   python check_gpu.py")
    else:
        print("\n✅ GPU ĐÃ SẴN SÀNG!")
        print("\n📝 Pipeline hiện tại SẼ TỰ ĐỘNG dùng GPU cho:")
        print("   • Whisper transcription (step1)")
        print("   • Pyannote diarization (step2)")
        
        print("\n⚡ Để tối ưu hiệu suất:")
        print("   • Đóng các app GPU khác (game, video rendering)")
        print("   • Theo dõi VRAM: nvidia-smi -l 1")
        print("   • Nếu out of memory, giảm batch size hoặc dùng model nhỏ hơn")


if __name__ == "__main__":
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "GPU DIAGNOSTIC TOOL" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Chạy các kiểm tra
    gpu_ok = check_gpu_detailed()
    
    if gpu_ok:
        test_gpu_computation()
    
    check_whisper_gpu()
    check_pyannote_gpu()
    get_nvidia_smi_info()
    show_recommendations()
    
    print("\n" + "=" * 80)
    print("✅ HOÀN THÀNH KIỂM TRA")
    print("=" * 80 + "\n")
