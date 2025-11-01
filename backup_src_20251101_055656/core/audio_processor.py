"""Audio preprocessing để cải thiện chất lượng."""
import numpy as np
import soundfile as sf
from scipy import signal
import os

def enhance_audio(input_path, output_path):
    """
    Cải thiện chất lượng audio:
    - Noise reduction (giảm nhiễu)
    - Normalize volume
    - Convert về 16kHz mono
    """
    print("🎵 Đang cải thiện chất lượng audio...")
    
    # ✅ FIX: Convert MP3/M4A/... sang WAV trước nếu cần
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext in ['.mp3', '.m4a', '.aac', '.ogg', '.flac']:
        print(f"   🔄 Phát hiện format {file_ext}, đang convert sang WAV...")
        from pydub import AudioSegment
        
        temp_wav = input_path.rsplit('.', 1)[0] + '_temp.wav'
        audio = AudioSegment.from_file(input_path)
        audio.export(temp_wav, format='wav')
        input_path = temp_wav
        print(f"   ✅ Đã convert sang: {temp_wav}")
    
    # Load audio
    waveform, sample_rate = sf.read(input_path)
    
    # Convert stereo → mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    
    # 1. Normalize volume
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.95  # Tránh clipping
    
    # 2. Resample về 16kHz (chuẩn cho ASR)
    if sample_rate != 16000:
        print(f"   🔄 Resample từ {sample_rate}Hz → 16000Hz...")
        num_samples = int(len(waveform) * 16000 / sample_rate)
        waveform = signal.resample(waveform, num_samples)
        sample_rate = 16000
    
    # 3. High-pass filter để giảm nhiễu tần số thấp
    sos = signal.butter(4, 80, 'hp', fs=sample_rate, output='sos')
    waveform = signal.sosfilt(sos, waveform)
    
    # 4. Normalize lại sau khi filter
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.95
    
    # Lưu kết quả
    sf.write(output_path, waveform, sample_rate)
    
    print(f"   ✅ Đã lưu audio enhanced: {output_path}")
    
    # Cleanup temp file nếu có
    if file_ext in ['.mp3', '.m4a', '.aac', '.ogg', '.flac']:
        try:
            os.remove(temp_wav)
        except:
            pass
    
    return output_path
