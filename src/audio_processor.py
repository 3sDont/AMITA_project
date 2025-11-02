"""
GIAI ĐOẠN 0: Audio preprocessing để cải thiện chất lượng.
- Convert sang WAV nếu cần
- Normalize volume
- Resample về 16kHz mono
- High-pass filter để giảm nhiễu
"""
import sys
import os
# Thêm thư mục gốc vào sys.path để import config và utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')  # Tắt warnings
import time
from datetime import datetime
import numpy as np
import soundfile as sf
from scipy import signal
from pydub import AudioSegment
import config


def enhance_audio(input_path, output_path=None):
    """
    Cải thiện chất lượng audio:
    - Convert sang WAV nếu cần
    - Normalize volume
    - Resample về 16kHz mono
    - High-pass filter để giảm nhiễu
    
    Args:
        input_path: Đường dẫn file audio đầu vào
        output_path: Đường dẫn file output (None = tạo tên tự động trong outputs/)
    
    Returns:
        str: Đường dẫn file đã xử lý
    """
    #start_time = time.time()
    #start_dt = datetime.now().strftime("%H:%M:%S")
    
    print("🎵 Đang cải thiện chất lượng audio...")
    #print(f"   ⏰ Bắt đầu lúc: {start_dt}")
    print(f"   📂 Input: {os.path.basename(input_path)}\n")
    
    # Tạo output path nếu không được cung cấp
    if output_path is None:
        os.makedirs("outputs", exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"outputs/{basename}_enhanced.wav"
    
    # Convert sang WAV nếu cần
    temp_wav = None
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext in ['.mp3', '.m4a', '.aac', '.ogg', '.flac']:
        print(f"   🔄 Convert {file_ext} → WAV...")
        convert_start = time.time()
        
        temp_wav = input_path.rsplit('.', 1)[0] + '_temp.wav'
        audio = AudioSegment.from_file(input_path)
        audio.export(temp_wav, format='wav')
        
        convert_time = time.time() - convert_start
        print(f"      ✅ Convert hoàn thành ({convert_time:.1f}s)")
        print(f"      📄 File: {os.path.basename(temp_wav)}\n")
        input_path = temp_wav
    
    # Load audio
    print("   📥 Đang load audio...")
    load_start = time.time()
    waveform, sample_rate = sf.read(input_path)
    #load_time = time.time() - load_start
    #print(f"      ✅ Load hoàn thành ({load_time:.1f}s)")
    print(f"      📊 Sample rate: {sample_rate}Hz")
    print(f"      📊 Channels: {waveform.ndim}")
    print(f"      📊 Duration: {len(waveform)/sample_rate:.1f}s\n")
    
    # Convert stereo → mono
    if waveform.ndim > 1:
        print("   🔄 Convert stereo → mono...")
        waveform = np.mean(waveform, axis=1)
        print("      ✅ Đã convert sang mono\n")
    
    # Normalize volume
    print("   🔊 Normalize volume...")
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.95
        print(f"      ✅ Đã normalize (max: {max_val:.3f} → 0.95)\n")
    
    # Resample về 16kHz
    if sample_rate != 16000:
        print(f"   🔄 Resample {sample_rate}Hz → 16000Hz...")
        resample_start = time.time()
        num_samples = int(len(waveform) * 16000 / sample_rate)
        waveform = signal.resample(waveform, num_samples)
        sample_rate = 16000
        resample_time = time.time() - resample_start
        print(f"      ✅ Resample hoàn thành ({resample_time:.1f}s)\n")
    
    # High-pass filter (loại bỏ nhiễu tần số thấp)
    print("   🎛️  Apply high-pass filter (80Hz)...")
    #filter_start = time.time()
    sos = signal.butter(4, 80, 'hp', fs=sample_rate, output='sos')
    waveform = signal.sosfilt(sos, waveform)
    #filter_time = time.time() - filter_start
    #print(f"      ✅ Filter hoàn thành ({filter_time:.1f}s)\n")
    
    # Normalize lại sau filter
    print("   🔊 Final normalize...")
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.95
        print(f"      ✅ Đã normalize final\n")
    
    # Lưu kết quả
    print("   💾 Đang lưu audio đã xử lý...")
    #save_start = time.time()
    sf.write(output_path, waveform, sample_rate)
    #save_time = time.time() - save_start
    #print(f"      ✅ Đã lưu ({save_time:.1f}s)")
    print(f"      📄 File: {os.path.basename(output_path)}\n")
    
    # Cleanup temp file
    if temp_wav and os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
            print("   🗑️  Đã xóa file temp\n")
        except Exception as e:
            print(f"   ⚠️  Không thể xóa temp file: {e}\n")
    
    #elapsed = time.time() - start_time
    #end_dt = datetime.now().strftime("%H:%M:%S")
    
    #print(f"⏱️  Thời gian preprocessing: {elapsed:.2f}s")
    #print(f"   ⏰ Kết thúc lúc: {end_dt}\n")
    
    return output_path


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🎵 GIAI ĐOẠN 0: AUDIO PREPROCESSING")
        print("="*80 + "\n")
        
        # Kiểm tra file audio có tồn tại không
        if not os.path.exists(config.AUDIO_FILE):
            print(f"❌ Lỗi: Không tìm thấy file audio: {config.AUDIO_FILE}")
            print(f"💡 Vui lòng kiểm tra lại đường dẫn trong config.py")
            sys.exit(1)
        
        # Tạo thư mục outputs nếu chưa có
        os.makedirs("outputs", exist_ok=True)
        
        # Tạo tên file output
        basename = os.path.splitext(os.path.basename(config.AUDIO_FILE))[0]
        output_file = f"outputs/{basename}_enhanced.wav"
        
        print(f"📁 Input:  {config.AUDIO_FILE}")
        print(f"📁 Output: {output_file}\n")
        
        # File info
        file_size_mb = os.path.getsize(config.AUDIO_FILE) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.1f} MB")
        
        # Chạy enhancement
        result = enhance_audio(config.AUDIO_FILE, output_file)
        
        print("="*80)
        print("✅ HOÀN THÀNH GIAI ĐOẠN 0")
        print("="*80)
        print(f"✓ Audio đã được xử lý và lưu tại: {result}")
        
        # File info output
        output_size_mb = os.path.getsize(result) / (1024 * 1024)
        print(f"\n📊 Thông tin file output:")
        print(f"   - Size: {output_size_mb:.1f} MB")
        print(f"   - Format: WAV")
        print(f"   - Sample rate: 16000 Hz")
        print(f"   - Channels: Mono")
        
        # Load để kiểm tra
        info = sf.info(result)
        print(f"   - Duration: {info.duration:.1f}s ({info.duration/60:.1f} phút)")
        
        print("\n" + "="*80)
        print("💡 TIP:")
        print("   Bây giờ bạn có thể dùng file này cho các giai đoạn tiếp theo:")
        print("   1. Cập nhật AUDIO_FILE trong config.py:")
        print(f"      AUDIO_FILE = r\"{os.path.abspath(result)}\"")
        print("   2. Hoặc chạy pipeline với file gốc (preprocessing optional)")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
