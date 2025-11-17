"""
Script kiểm tra thông tin audio file
"""
import librosa
import soundfile as sf
from config import AUDIO_FILE


def check_audio_info_librosa(audio_path):
    """Kiểm tra thông tin audio bằng librosa"""
    print("=" * 70)
    print("🎵 THÔNG TIN AUDIO - LIBROSA")
    print("=" * 70)
    
    # Load audio và lấy sample rate
    y, sr = librosa.load(audio_path, sr=None)  # sr=None để giữ nguyên sample rate gốc
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"📁 File: {audio_path}")
    print(f"🎼 Sample Rate: {sr} Hz")
    print(f"⏱️  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"📊 Number of Samples: {len(y):,}")
    print(f"📏 Shape: {y.shape}")
    print(f"🔢 Data Type: {y.dtype}")
    print(f"📈 Min Value: {y.min():.6f}")
    print(f"📉 Max Value: {y.max():.6f}")
    print("=" * 70)
    
    return sr, duration


def check_audio_info_soundfile(audio_path):
    """Kiểm tra thông tin audio bằng soundfile (nhanh hơn)"""
    print("\n" + "=" * 70)
    print("🎵 THÔNG TIN AUDIO - SOUNDFILE")
    print("=" * 70)
    
    info = sf.info(audio_path)
    
    print(f"📁 File: {audio_path}")
    print(f"🎼 Sample Rate: {info.samplerate} Hz")
    print(f"⏱️  Duration: {info.duration:.2f} seconds ({info.duration/60:.2f} minutes)")
    print(f"📊 Number of Frames: {info.frames:,}")
    print(f"🔊 Channels: {info.channels} ({'Mono' if info.channels == 1 else 'Stereo'})")
    print(f"📦 Format: {info.format}")
    print(f"🎚️  Subtype: {info.subtype}")
    print("=" * 70)
    
    return info.samplerate, info.duration


def check_audio_quick(audio_path):
    """Chỉ lấy sample rate nhanh nhất"""
    info = sf.info(audio_path)
    return info.samplerate


if __name__ == "__main__":
    print("\n🔍 KIỂM TRA AUDIO FILE\n")
    
    try:
        # Cách 1: Soundfile (nhanh, không load dữ liệu)
        sr_sf, duration_sf = check_audio_info_soundfile(AUDIO_FILE)
        
        # Cách 2: Librosa (chậm hơn, load toàn bộ audio)
        print("\n⚠️  Đang load audio bằng librosa (có thể mất vài giây)...")
        sr_lib, duration_lib = check_audio_info_librosa(AUDIO_FILE)
        
        # So sánh
        print("\n" + "=" * 70)
        print("📊 SO SÁNH KẾT QUẢ")
        print("=" * 70)
        print(f"Sample Rate khớp: {'✅ YES' if sr_sf == sr_lib else '❌ NO'}")
        print(f"Duration khớp: {'✅ YES' if abs(duration_sf - duration_lib) < 0.1 else '❌ NO'}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("\n💡 Đảm bảo file audio tồn tại tại đường dẫn trong config.py")
