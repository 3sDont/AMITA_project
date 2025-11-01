"""Audio quality debugging tool."""
import soundfile as sf
import numpy as np
import sys

def analyze_audio(audio_path):
    """Phân tích audio để phát hiện vấn đề."""
    print(f"🔍 Phân tích file: {audio_path}\n")
    
    # Load audio
    waveform, sr = sf.read(audio_path)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    
    duration = len(waveform) / sr
    
    print(f"📊 Thông tin cơ bản:")
    print(f"   - Sample rate: {sr} Hz")
    print(f"   - Độ dài: {duration:.1f}s ({duration/60:.1f} phút)")
    print(f"   - Số samples: {len(waveform):,}")
    
    # Phân tích âm lượng
    rms = np.sqrt(np.mean(waveform**2))
    peak = np.max(np.abs(waveform))
    
    print(f"\n🔊 Âm lượng:")
    print(f"   - RMS: {rms:.4f}")
    print(f"   - Peak: {peak:.4f}")
    print(f"   - Dynamic range: {20*np.log10(peak/rms):.1f} dB")
    
    # Phát hiện im lặng
    silence_threshold = 0.01
    silence_samples = np.sum(np.abs(waveform) < silence_threshold)
    silence_ratio = silence_samples / len(waveform)
    
    print(f"\n🤫 Im lặng:")
    print(f"   - Tỷ lệ im lặng: {silence_ratio*100:.1f}%")
    print(f"   - Thời gian im lặng: {silence_ratio*duration:.1f}s")
    
    # Phát hiện đoạn lặp (watermark)
    chunk_size = sr * 10  # 10s chunks
    chunks = [waveform[i:i+chunk_size] for i in range(0, len(waveform), chunk_size)]
    
    if len(chunks) > 3:
        # So sánh 3 chunks đầu tiên
        correlations = []
        for i in range(len(chunks)-1):
            if len(chunks[i]) == len(chunks[i+1]):
                corr = np.corrcoef(chunks[i], chunks[i+1])[0,1]
                correlations.append(corr)
        
        if correlations:
            avg_corr = np.mean(correlations)
            print(f"\n🔁 Phát hiện lặp:")
            print(f"   - Tương quan giữa các chunks: {avg_corr:.3f}")
            if avg_corr > 0.9:
                print("   ⚠️  CẢNH BÁO: Audio có vẻ chứa đoạn lặp (watermark?)")
    
    # Phân tích phổ tần
    fft = np.fft.rfft(waveform[:sr*10])  # FFT của 10s đầu
    freqs = np.fft.rfftfreq(sr*10, 1/sr)
    magnitude = np.abs(fft)
    
    dominant_freq = freqs[np.argmax(magnitude)]
    
    print(f"\n🎵 Phổ tần:")
    print(f"   - Tần số mạnh nhất: {dominant_freq:.1f} Hz")
    
    # Đánh giá chung
    print(f"\n📋 Đánh giá:")
    issues = []
    
    if silence_ratio > 0.5:
        issues.append("⚠️  Quá nhiều im lặng (>50%)")
    if rms < 0.01:
        issues.append("⚠️  Âm lượng quá nhỏ")
    if peak < 0.1:
        issues.append("⚠️  Không có đỉnh rõ ràng")
    
    if issues:
        print("   Vấn đề phát hiện:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ Audio có vẻ ổn")
    
    print("\n💡 Khuyến nghị:")
    if silence_ratio > 0.3:
        print("   - Cân nhắc cắt bỏ đoạn im lặng dài")
    if rms < 0.01:
        print("   - Cân nhắc tăng âm lượng (normalize)")
    
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
