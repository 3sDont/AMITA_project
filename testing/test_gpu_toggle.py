"""
Script test GPU toggle - so sánh thời gian CPU vs GPU
"""
import time
import torch
import config


def test_gpu_toggle():
    """Test bật tắt GPU"""
    print("=" * 80)
    print("🧪 TEST BẬT TẮT GPU")
    print("=" * 80)
    
    print(f"\n📋 Config hiện tại:")
    print(f"   • USE_GPU = {config.USE_GPU}")
    print(f"   • CUDA available = {torch.cuda.is_available()}")
    
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   • Device sẽ dùng: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"   • Device sẽ dùng: CPU")
    
    print(f"\n💡 Để thay đổi:")
    print(f"   1. Mở file config.py")
    print(f"   2. Tìm dòng: USE_GPU = {config.USE_GPU}")
    print(f"   3. Đổi thành: USE_GPU = {not config.USE_GPU}")
    print(f"   4. Lưu file và chạy lại")
    
    # Demo tensor operation
    print(f"\n🔄 Demo: Tạo tensor trên {device.type.upper()}...")
    try:
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        start = time.time()
        z = torch.matmul(x, y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        
        print(f"✅ Tính toán ma trận 1000x1000 trên {device.type.upper()}: {elapsed:.2f}ms")
        
        if device.type == "cuda":
            vram_used = torch.cuda.memory_allocated() / (1024**2)
            print(f"💾 VRAM đang dùng: {vram_used:.2f} MB")
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_gpu_toggle()
    
    print("\n📝 HƯỚNG DẪN SỬ DỤNG:")
    print("-" * 80)
    print("\n1️⃣  CHẠY VỚI GPU (nhanh, yêu cầu NVIDIA GPU):")
    print("   • Mở config.py")
    print("   • Đặt: USE_GPU = True")
    print("   • Chạy: python main.py")
    
    print("\n2️⃣  CHẠY VỚI CPU (chậm, nhưng chạy trên mọi máy):")
    print("   • Mở config.py")
    print("   • Đặt: USE_GPU = False")
    print("   • Chạy: python main.py")
    
    print("\n3️⃣  AUTO FALLBACK:")
    print("   • Nếu USE_GPU = True nhưng không có GPU")
    print("   • Hệ thống tự động chuyển về CPU")
    print("   • Không cần thay đổi code")
    
    print("\n⚡ SO SÁNH TỐC ĐỘ (ước tính):")
    print("   • Whisper medium trên audio 3 phút:")
    print("     - GPU: ~30-60 giây")
    print("     - CPU: ~5-10 phút")
    print("   • Pyannote diarization:")
    print("     - GPU: ~10-20 giây")
    print("     - CPU: ~1-3 phút")
    
    print("\n💡 KHUYẾN NGHỊ:")
    print("   • Có GPU → USE_GPU = True (nhanh hơn 5-10x)")
    print("   • Không GPU → USE_GPU = False")
    print("   • Test lần đầu → Để True, để hệ thống tự fallback")
    print("\n" + "=" * 80)
