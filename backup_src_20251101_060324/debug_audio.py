"""Debug tool để kiểm tra audio quality."""
import numpy as np
import soundfile as sf
import sys


def analyze_audio(audio_path):
    """
    Phân tích audio để phát hiện vấn đề.
    
    Returns:
        dict: Thống kê audio
    """
    print(f"🔍 Phân tích: {audio_path}\n")
    
    # Load audio
    waveform, sr = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    
    duration = len(waveform) / sr
    
    print(f"📊 Thông tin:")
    print(f"   - Sample rate: {sr} Hz")
    print(f"   - Độ dài: {duration:.1f}s ({duration/60:.1f} phút)")
    print(f"   - Số samples: {len(waveform):,}")
    
    # Âm lượng
    rms = np.sqrt(np.mean(waveform**2))
    peak = np.max(np.abs(waveform))
    
    print(f"\n🔊 Âm lượng:")
    print(f"   - RMS: {rms:.4f}")
    print(f"   - Peak: {peak:.4f}")
    
    if peak > 0:
        print(f"   - Dynamic range: {20*np.log10(peak/rms):.1f} dB")
    
    # Im lặng
    silence_threshold = 0.01
    silence_samples = np.sum(np.abs(waveform) < silence_threshold)
    silence_ratio = silence_samples / len(waveform)
    
    print(f"\n🤫 Im lặng:")
    print(f"   - Tỷ lệ: {silence_ratio*100:.1f}%")
    print(f"   - Thời gian: {silence_ratio*duration:.1f}s")
    
    # Đánh giá
    print(f"\n📋 Đánh giá:")
    issues = []
    
    if silence_ratio > 0.5:
        issues.append("⚠️  Quá nhiều im lặng (>50%)")
    if rms < 0.01:
        issues.append("⚠️  Âm lượng quá nhỏ")
    if peak < 0.1:
        issues.append("⚠️  Không có đỉnh rõ ràng")
    
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ Audio ổn")
    
    print("\n💡 Khuyến nghị:")
    if silence_ratio > 0.3:
        print("   - Cắt bỏ đoạn im lặng dài")
    if rms < 0.01:
        print("   - Tăng âm lượng (normalize)")
    
    return {
        "duration": duration,
        "silence_ratio": silence_ratio,
        "rms": rms,
        "peak": peak
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_audio.py <audio_file>")
        sys.exit(1)
    
    analyze_audio(sys.argv[1])
