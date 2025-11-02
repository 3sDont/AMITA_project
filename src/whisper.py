"""
Optimized transcription với faster-whisper.
Nhanh gấp 4x, dùng ít RAM hơn 50% so với openai-whisper.
"""
import sys
import os
# Thêm thư mục gốc vào sys.path để import config và utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')  # Tắt warnings
import time
import os
from datetime import datetime
import soundfile as sf
from pydub import AudioSegment
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
import config
import torch
import utils


def is_similar_text(text1, text2, threshold=0.85):
    """Kiểm tra 2 text có giống nhau không."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > threshold


def get_device():
    """Xác định device (GPU/CPU) dựa trên config"""
    if config.USE_GPU and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 Sử dụng GPU: {gpu_name}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   VRAM: {vram:.2f} GB")
    else:
        device = "cpu"
        if config.USE_GPU and not torch.cuda.is_available():
            print(f"⚠️  GPU không khả dụng, fallback về CPU")
        else:
            print(f"🖥️  Sử dụng CPU (USE_GPU=False)")
    return device

def split_audio_simple(audio_path, chunk_length_minutes=10):
    """Fallback: Chia audio đơn giản theo thời gian."""
    info = sf.info(audio_path)
    total_duration_minutes = info.duration / 60
    
    if total_duration_minutes <= chunk_length_minutes:
        print(f"   ℹ️  File ngắn ({total_duration_minutes:.1f} phút), không cần chia")
        return [(audio_path, 0, info.duration)]
    
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = chunk_length_minutes * 60 * 1000
    
    chunks = []
    base_name = os.path.splitext(audio_path)[0]
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        start_time = i / 1000
        end_time = min((i + chunk_length_ms) / 1000, len(audio) / 1000)
        
        chunk_path = f"{base_name}_chunk_{i//chunk_length_ms:03d}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_time, end_time))
    
    print(f"   ✅ Đã chia thành {len(chunks)} chunks (theo thời gian)")
    return chunks

def split_audio_by_segments(audio_path, chunk_length_minutes=10):
    """
    Chia audio thành chunks nhưng kết thúc tại segment boundaries.
    
    Args:
        audio_path: Đường dẫn file audio
        diarization_path: File diarization để lấy segment boundaries (nếu None, dùng config.DIARIZATION_CACHE)
        chunk_length_minutes: Độ dài mục tiêu mỗi chunk (phút)
    
    Returns:
        List of (chunk_path, start_time, end_time) tuples
    """
    print(f"   ✂️  Chia audio thông minh theo segment boundaries...")
    
    # Xác định file diarization
    diar_file = config.DIARIZATION_CACHE
    
    if not os.path.exists(diar_file):
        print(f"   ⚠️  Không tìm thấy file diarization: {diar_file}")
        print(f"   ⚠️  Chia thông thường...")
        return split_audio_simple(audio_path, chunk_length_minutes)
    
    segments = utils.load_json(diar_file)
    if not segments:
        print("   ⚠️  Không có segments, chia thông thường...")
        return split_audio_simple(audio_path, chunk_length_minutes)
    
    # Load audio info
    info = sf.info(audio_path)
    total_duration = info.duration
    chunk_length_seconds = chunk_length_minutes * 60
    
    print(f"   📊 Tổng thời gian: {total_duration/60:.1f} phút")
    print(f"   📊 Tổng số segments: {len(segments)}")
    
    # Tạo chunk boundaries
    chunks = []
    current_start = 0
    chunk_index = 0
    
    audio = AudioSegment.from_file(audio_path)
    base_name = os.path.splitext(audio_path)[0]
    
    while current_start < total_duration:
        # Tìm segment cuối cùng trước chunk_length_seconds
        target_end = current_start + chunk_length_seconds
        
        # Tìm segment gần nhất với target_end
        best_seg = None
        for seg in segments:
            if seg['start_time'] >= current_start and seg['end_time'] <= total_duration:
                # Nếu segment này gần target_end nhất
                if seg['end_time'] <= target_end + 60:  # Cho phép sai lệch 60s
                    best_seg = seg
                elif seg['start_time'] > target_end:
                    break  # Đã quá xa
        
        # Xác định điểm cắt
        if best_seg:
            chunk_end = best_seg['end_time']
        else:
            # Không tìm được segment phù hợp, dùng target_end
            chunk_end = min(target_end, total_duration)
        
        # Tránh chunk quá ngắn (< 30s)
        if chunk_end - current_start < 30 and current_start > 0:
            print(f"   ⚠️  Chunk quá ngắn ({chunk_end - current_start:.1f}s), gộp vào chunk trước")
            # Gộp vào chunk trước
            if chunks:
                last_chunk_path, _, _ = chunks[-1]
                chunks[-1] = (last_chunk_path, chunks[-1][1], chunk_end)
            break
        
        # Extract chunk
        start_ms = int(current_start * 1000)
        end_ms = int(chunk_end * 1000)
        chunk_audio = audio[start_ms:end_ms]
        
        chunk_path = f"{base_name}_chunk_{chunk_index:03d}.wav"
        chunk_audio.export(chunk_path, format="wav")
        
        chunks.append((chunk_path, current_start, chunk_end))
        
        print(f"   ✅ Chunk {chunk_index + 1}: [{current_start:.1f}s - {chunk_end:.1f}s] ({chunk_end - current_start:.1f}s)")
        
        current_start = chunk_end
        chunk_index += 1
        
        # Tránh vòng lặp vô hạn
        if chunk_index > 1000:
            print("   ⚠️  Quá nhiều chunks, dừng lại")
            break
    
    print(f"   ✅ Đã chia thành {len(chunks)} chunks (theo segment boundaries)")
    return chunks


def transcribe_chunk_optimized(model, chunk_path, chunk_index, total_chunks):
    """Transcribe 1 chunk với faster-whisper."""
    print(f"\n   📝 Xử lý chunk {chunk_index + 1}/{total_chunks}...")
    
    # ✅ faster-whisper API khác với openai-whisper
    segments, info = model.transcribe(
        chunk_path,
        language="vi",
        beam_size=5,
        vad_filter=True,  # ✅ Voice Activity Detection tự động
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Tối thiểu 500ms im lặng để cắt
            threshold=0.5,                # Độ nhạy phát hiện giọng nói
        ),
        word_timestamps=True,  # Timestamps từng từ
        condition_on_previous_text=True,
    )
    
    # Convert generator to list
    segments_list = []
    for segment in segments:
        segments_list.append({
            "start_time": segment.start,  # ✅ faster-whisper dùng .start không phải .start_time
            "end_time": segment.end,      # ✅ faster-whisper dùng .end không phải .end_time
            "text": segment.text,
            "no_speech_prob": segment.no_speech_prob,
            "words": [
                {"word": w.word, "start_time": w.start, "end_time": w.end, "probability": w.probability}
                for w in (segment.words or [])
            ] if segment.words else []
        })
    
    print(f"      ✅ Chunk {chunk_index + 1}: {len(segments_list)} segments")
    return {"segments": segments_list, "language": info.language}


def transcribe_audio_optimized(audio_path,  chunk_length_minutes=10):
    """
    Transcription tối ưu với faster-whisper.
    """
    start_time = time.time()
    print("🎧 Đang chạy Whisper transcription (OPTIMIZED)...")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {audio_path}")
    
    device = get_device()
    
    # ✅ Load model với progress
    print("   📥 Đang load Whisper model 'medium' (faster-whisper)...")
    load_start = time.time()
    
    # Xác định compute_type dựa trên device
    if device == "cuda":
        compute_type = "float16"  # GPU dùng float16 cho tốc độ tốt nhất
    else:
        compute_type = "int8"  # CPU dùng int8 cho hiệu suất tốt
    
    model = WhisperModel(
        "medium",
        device=device,
        compute_type=compute_type,
        cpu_threads=4 if device == "cpu" else 1,
        num_workers=1,
    )
    load_time = time.time() - load_start
    print(f"   ✅ Load model: {load_time:.1f}s (device={device}, compute_type={compute_type})")
    
    
    print(f"   🎵 Đang xử lý file: {os.path.basename(audio_path)}")
    
    info = sf.info(audio_path)
    duration_minutes = info.duration / 60
    print(f"   ⏱️  Độ dài audio: {duration_minutes:.1f} phút")
    
    # ✅ Chia chunks với progress
    chunk_split_start = time.time()
    if duration_minutes > chunk_length_minutes:
        print(f"   📌 File dài ({duration_minutes:.1f} phút) → Chunked processing")
        if os.path.exists(config.DIARIZATION_CACHE):
            print(f"   🎯 Chia chunks theo diarization segments từ cache: {os.path.basename(config.DIARIZATION_CACHE)}")
            chunks = split_audio_by_segments(audio_path, chunk_length_minutes)
        else:
            print("   ⚠️  Không có diarization, chia chunks theo thời gian...")
            chunks = split_audio_simple(audio_path, chunk_length_minutes)
    else:
        print(f"   ℹ️  File ngắn ({duration_minutes:.1f} phút), không cần chia chunks")
        chunks = [(audio_path, 0, info.duration)]
    
    chunk_time = time.time() - chunk_split_start
    print(f"   ✅ Chia chunks: {chunk_time:.1f}s\n")
    
    # Transcribe từng chunk với ETA
    trans_start = time.time()
    all_segments = []
    
    for idx, (chunk_path, chunk_start_time, chunk_end) in enumerate(chunks):
        chunk_duration = chunk_end - chunk_start_time
        print(f"\n   📝 Xử lý chunk {idx+1}/{len(chunks)} ({chunk_duration:.1f}s)...")
        
        chunk_trans_start = time.time()
        chunk_result = transcribe_chunk_optimized(model, chunk_path, idx, len(chunks))
        chunk_trans_time = time.time() - chunk_trans_start
        
        # ✅ ETA cho các chunks còn lại
        if idx < len(chunks) - 1:
            avg_time_per_chunk = (time.time() - trans_start) / (idx + 1)
            remaining_chunks = len(chunks) - idx - 1
            eta = avg_time_per_chunk * remaining_chunks
            print(f"      ⏱️  Chunk time: {chunk_trans_time:.1f}s | ETA: {eta:.1f}s ({eta/60:.1f} min)")
        
        # Điều chỉnh timestamps
        for seg in chunk_result.get("segments", []):
            seg["start_time"] += chunk_start_time
            seg["end_time"] += chunk_start_time
            all_segments.append(seg)
        
        # Cleanup
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except:
                pass
    
    trans_time = time.time() - trans_start
    print(f"\n   ✅ Transcription hoàn thành: {trans_time:.1f}s\n")
    
    # Clear GPU cache nếu dùng GPU
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"   ✅ Tổng cộng phát hiện {len(all_segments)} segments")
    
    # ✅ Lọc spam với progress
    print("   🔍 Đang lọc spam, noise và repeats...")
    filter_start = time.time()
    
    filtered_segments = []
    spam_keywords = [
        "đăng ký kênh", "ủng hộ kênh", "subscribe", "like share",
        "theo dõi", "bật chuông", "anh cứ cầm đi",
    ]
    
    spam_count = 0
    silence_count = 0
    repeat_count = 0
    short_count = 0
    
    prev_texts = []
    window_size = 5
    
    for seg in all_segments:
        text = seg.get("text", "").strip()
        no_speech_prob = seg.get("no_speech_prob", 0)
        
        if not text:
            continue
        
        if no_speech_prob > 0.9:
            silence_count += 1
            continue
        
        if any(kw in text.lower() for kw in spam_keywords):
            spam_count += 1
            continue
        
        # Phát hiện lặp
        is_repeat = False
        for prev_text in prev_texts[-window_size:]:
            if is_similar_text(text, prev_text, threshold=0.85):
                repeat_count += 1
                is_repeat = True
                break
        if is_repeat:
            continue
        
        # Bỏ qua segments quá ngắn
        duration = seg["end_time"] - seg["start_time"]
        if duration < 0.3:
            short_count += 1
            continue
        
        filtered_segments.append(seg)
        prev_texts.append(text)
        if len(prev_texts) > window_size * 2:
            prev_texts.pop(0)
    
    filter_time = time.time() - filter_start
    print(f"   ✅ Lọc hoàn thành: {filter_time:.1f}s\n")
    
    print(f"   🗑️  Đã loại bỏ: {spam_count} spam, {silence_count} im lặng, {repeat_count} repeats, {short_count} quá ngắn")
    print(f"   📊 Còn lại: {len(filtered_segments)} segments")
    
    
    output = {
        "text": " ".join(seg.get("text", "") for seg in filtered_segments),
        "segments": filtered_segments,
        "language": "vi"
    }
    
    
    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n⏱️  THỜI GIAN CHI TIẾT:")
    print(f"   - Load model:       {load_time:>6.1f}s")
    print(f"   - Chia chunks:      {chunk_time:>6.1f}s")
    print(f"   - Transcription:    {trans_time:>6.1f}s")
    print(f"   - Filter:           {filter_time:>6.1f}s")
    print(f"   {'─'*35}")
    print(f"   🕓 TOTAL:           {elapsed:>6.1f}s ({elapsed/60:.1f} phút)")
    print(f"   ⏰ Kết thúc lúc: {end_dt}")
    
    total_text_length = sum(len(seg.get("text", "")) for seg in filtered_segments)
    print(f"\n   📝 Tổng số ký tự: {total_text_length:,}")
    if elapsed > 0:
        speedup = duration_minutes / (elapsed / 60)
        print(f"   🎯 Tốc độ xử lý: {speedup:.2f}x realtime")
    
    # Lưu cache
    utils.save_json(output, config.WHISPER_CACHE)
    return output


if __name__ == "__main__":
    """Chạy riêng giai đoạn 1: Whisper Transcription"""
    import sys
    
    try:
        print("\n" + "="*80)
        print("🎤 GIAI ĐOẠN 1: WHISPER TRANSCRIPTION (OPTIMIZED)")
        print("="*80 + "\n")
        
        # Kiểm tra file audio có tồn tại không
        if not os.path.exists(config.AUDIO_FILE):
            print(f"❌ Lỗi: Không tìm thấy file audio: {config.AUDIO_FILE}")
            print(f"💡 Vui lòng kiểm tra lại đường dẫn trong config.py")
            sys.exit(1)
        

        
        # Tạo thư mục outputs nếu chưa có
        os.makedirs("outputs", exist_ok=True)
        
        print(f"📁 Input:  {config.AUDIO_FILE}")
        print(f"⚙️  Config: USE_GPU={config.USE_GPU}\n")
        
        # Chạy transcription
        result = transcribe_audio_optimized(
            audio_path=config.AUDIO_FILE,
            chunk_length_minutes=10
        )

        print(f"\n✓ Kết quả đã lưu tại: {config.WHISPER_CACHE}")
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH GIAI ĐOẠN 1")
        print("="*80)
        print(f"📊 Số segments: {len(result['segments'])}")
        print(f"📝 Tổng text: {len(result['text'])} ký tự")
        print(f"🗣️  Ngôn ngữ: {result['language']}")
        
        # Preview một số segments đầu
        print(f"\n📋 Preview 3 segments đầu tiên:")
        print("-" * 80)
        for i, seg in enumerate(result['segments'][:3]):
            print(f"\n[{i+1}] {seg['start_time']:.2f}s → {seg['end_time']:.2f}s")
            print(f"    Text: {seg['text'][:100]}{'...' if len(seg['text']) > 100 else ''}")
        
        if len(result['segments']) > 3:
            print(f"\n... và {len(result['segments']) - 3} segments khác")
        
        print("\n" + "="*80)
        print("💡 TIP: Chạy tiếp giai đoạn 2 (diarization) nếu chưa có:")
        print("   python src/diarization.py")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
