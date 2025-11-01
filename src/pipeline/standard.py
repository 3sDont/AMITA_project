"""Standard pipeline implementation."""
import sys
import time
import os
from datetime import datetime
import argparse
from pathlib import Path

# ✅ FIX: Add parent directory to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_utils import prepare_output_paths
from core.diarization import diarize_audio
from core.transcription import transcribe_audio
from core.gender_classifier import classify_gender
from core.combiner import combine_results
from core.audio_processor import enhance_audio




def convert_to_audio(input_path):
    """Convert video sang audio nếu cần."""
    from pydub import AudioSegment
    import os
    
    ext = os.path.splitext(input_path)[1].lower()
    
    # ✅ FIX: Xử lý cả audio formats khác MP3
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
        print(f"🎬 Phát hiện file video ({ext}), đang convert sang audio...")
        output_path = input_path.rsplit('.', 1)[0] + '_audio.wav'
        
        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format='wav')
            print(f"✅ Đã convert sang: {output_path}\n")
            return output_path
        except Exception as e:
            print(f"❌ Lỗi convert video: {e}")
            raise
    
    elif ext in ['.mp3', '.m4a', '.aac', '.ogg', '.flac']:
        print(f"🎵 Phát hiện file audio ({ext}), giữ nguyên để xử lý sau...")
        # Không convert ở đây, để enhance_audio xử lý
        return input_path
    
    elif ext == '.wav':
        print(f"✅ File WAV, không cần convert")
        return input_path
    
    else:
        print(f"⚠️  Format không nhận diện ({ext}), thử xử lý như audio...")
        return input_path


def main(audio_path, num_speakers=None, min_speakers=None, max_speakers=None, chunk_minutes=10):
    """Run standard pipeline."""
    total_start = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU PIPELINE")
    print(f"📅 Thời gian bắt đầu: {start_datetime}")
    print(f"📁 File input: {audio_path}")
    print(f"{'='*60}\n")
    
    # Convert video sang audio nếu cần
    audio_path = convert_to_audio(audio_path)
    
    # ✅ Tiền xử lý audio (sẽ tự động convert MP3→WAV)
    print("🔧 Tiền xử lý audio để cải thiện chất lượng...")
    # ✅ FIX: Dùng tên file gốc để tạo enhanced path
    base_name = os.path.splitext(audio_path)[0]
    enhanced_path = base_name + '_enhanced.wav'
    audio_path = enhance_audio(audio_path, enhanced_path)
    
    paths = prepare_output_paths(audio_path)

    # 1️⃣ Diarization với tham số tùy chỉnh
    step1_start = time.time()
    diarize_audio(
        audio_path, 
        paths["diarization"],
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    step1_time = time.time() - step1_start
    print(f"⏱️  Bước 1 hoàn thành trong: {step1_time:.2f}s ({step1_time/60:.1f} phút)\n")

    # 2️⃣ Transcription với chunk size custom và diarization
    step2_start = time.time()
    # ✅ THÊM: Truyền diarization_path để chia chunks thông minh
    transcribe_audio(
        audio_path, 
        paths["transcription"], 
        chunk_length_minutes=chunk_minutes,
        diarization_path=paths["diarization"]  # ← Thêm dòng này
    )
    step2_time = time.time() - step2_start
    print(f"⏱️  Bước 2 hoàn thành trong: {step2_time:.2f}s ({step2_time/60:.1f} phút)\n")

    # 3️⃣ Gender Classification
    step3_start = time.time()
    # ✅ THÊM: Truyền audio_path để phân tích thật
    classify_gender(paths["diarization"], paths["gender"], audio_path=audio_path)
    step3_time = time.time() - step3_start
    print(f"⏱️  Bước 3 hoàn thành trong: {step3_time:.2f}s\n")

    # 4️⃣ Combine
    step4_start = time.time()
    combine_results(paths["diarization"], paths["transcription"], paths["gender"], paths["combined"])
    step4_time = time.time() - step4_start
    print(f"⏱️  Bước 4 hoàn thành trong: {step4_time:.2f}s\n")

    total_elapsed = time.time() - total_start
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print("🎉 HOÀN TẤT PIPELINE")
    print(f"📅 Thời gian kết thúc: {end_datetime}")
    print(f"📂 Kết quả lưu trong: {paths['output_dir']}")
    print(f"\n📊 THỐNG KÊ THỜI GIAN:")
    print(f"   1️⃣  Diarization:        {step1_time:>8.2f}s ({step1_time/60:>5.1f} phút)")
    print(f"   2️⃣  Transcription:      {step2_time:>8.2f}s ({step2_time/60:>5.1f} phút)")
    print(f"   3️⃣  Gender classify:    {step3_time:>8.2f}s")
    print(f"   4️⃣  Combine results:    {step4_time:>8.2f}s")
    print(f"   {'─'*50}")
    print(f"   🕓 TỔNG THỜI GIAN:      {total_elapsed:>8.2f}s ({total_elapsed/60:>5.1f} phút)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard Pipeline")
    parser.add_argument("audio_path", type=str)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--chunk-minutes", type=int, default=10)
    
    args = parser.parse_args()
    
    main(
        args.audio_path,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        chunk_minutes=args.chunk_minutes
    )
