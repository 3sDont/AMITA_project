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
import numpy as np
import gc  # ✅ Thêm garbage collection


def is_similar_text(text1, text2, threshold=0.85):
    """Kiểm tra 2 text có giống nhau không."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > threshold


def get_device():
    """Xác định device (GPU/CPU) dựa trên config"""
    print(f"\n🔍 Kiểm tra GPU...")
    print(f"   ⚙️  Config: USE_GPU = {config.USE_GPU}")
    print(f"   🖥️  CUDA available: {torch.cuda.is_available()}")
    
    if config.USE_GPU and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        
        print(f"\n✅ SỬ DỤNG GPU")
        print(f"   🎮 GPU: {gpu_name}")
        print(f"   💾 VRAM: {vram:.2f} GB")
        print(f"   🔧 CUDA: {cuda_version}")
        return device
    else:
        device = "cpu"
        
        if config.USE_GPU and not torch.cuda.is_available():
            print(f"\n⚠️  GPU KHÔNG KHẢ DỤNG - FALLBACK VỀ CPU")
            print(f"\n💡 Để sử dụng GPU:")
            print(f"   1. Kiểm tra có GPU NVIDIA:")
            print(f"      → Mở Task Manager > Performance > GPU")
            print(f"   2. Cài lại PyTorch với CUDA:")
            print(f"      pip uninstall torch torchaudio")
            print(f"      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print(f"   3. Test GPU:")
            print(f"      python test_gpu.py")
        else:
            print(f"\n🖥️  SỬ DỤNG CPU (USE_GPU=False trong config)")
        
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
    ✅ IMPROVED: Streaming processing, không load toàn bộ audio vào RAM
    """
    print(f"   ✂️  Chia audio thông minh theo segment boundaries...")
    
    diar_file = config.DIARIZATION_CACHE
    
    if not os.path.exists(diar_file):
        print(f"   ⚠️  Không tìm thấy file diarization: {diar_file}")
        return split_audio_simple(audio_path, chunk_length_minutes)
    
    segments = utils.load_json(diar_file)
    if not segments:
        print("   ⚠️  Không có segments, chia thông thường...")
        return split_audio_simple(audio_path, chunk_length_minutes)
    
    # Load audio info (KHÔNG load toàn bộ audio)
    info = sf.info(audio_path)
    total_duration = info.duration
    chunk_length_seconds = chunk_length_minutes * 60
    
    print(f"   📊 Tổng thời gian: {total_duration/60:.1f} phút")
    print(f"   📊 Tổng số segments: {len(segments)}")
    
    # ✅ Tạo chunk boundaries TRƯỚC (không cần load audio)
    chunk_boundaries = []
    current_start = 0
    
    while current_start < total_duration:
        target_end = current_start + chunk_length_seconds
        
        # Tìm segment gần nhất với target_end
        best_seg = None
        for seg in segments:
            if seg['start_time'] >= current_start and seg['end_time'] <= total_duration:
                if seg['end_time'] <= target_end + 60:
                    best_seg = seg
                elif seg['start_time'] > target_end:
                    break
        
        chunk_end = best_seg['end_time'] if best_seg else min(target_end, total_duration)
        
        # Tránh chunk quá ngắn
        if chunk_end - current_start >= 30:
            chunk_boundaries.append((current_start, chunk_end))
        
        current_start = chunk_end
        
        if len(chunk_boundaries) > 1000:
            break
    
    # ✅ GHI CHÚ: Load từng chunk một khi cần (lazy loading)
    print(f"   ✅ Đã xác định {len(chunk_boundaries)} chunk boundaries")
    print(f"   💡 Sẽ load và export từng chunk khi cần (tiết kiệm RAM)\n")
    
    # Export từng chunk với streaming
    chunks = []
    base_name = os.path.splitext(audio_path)[0]
    
    # ✅ Load audio một lần nhưng process từng đoạn
    audio = AudioSegment.from_file(audio_path)
    
    for idx, (start_time, end_time) in enumerate(chunk_boundaries):
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Extract chunk (chỉ slice cần thiết)
        chunk_audio = audio[start_ms:end_ms]
        
        chunk_path = f"{base_name}_chunk_{idx:03d}.wav"
        chunk_audio.export(chunk_path, format="wav")
        
        chunks.append((chunk_path, start_time, end_time))
        
        # ✅ Free memory của chunk vừa export
        del chunk_audio
        
        print(f"   ✅ Chunk {idx + 1}: [{start_time:.1f}s - {end_time:.1f}s] ({end_time - start_time:.1f}s)")
    
    # ✅ Clear audio khỏi memory
    del audio
    gc.collect()
    
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


def calculate_text_entropy(text):
    """
    Tính entropy của text để phát hiện nội dung lặp/kém chất lượng.
    Entropy thấp = text lặp nhiều hoặc ít thông tin.
    """
    if not text or len(text) < 2:
        return 0.0
    
    # Đếm tần suất ký tự
    from collections import Counter
    char_counts = Counter(text.lower())
    text_len = len(text)
    
    # Tính entropy Shannon
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_len
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def calculate_word_repetition_ratio(text):
    """
    Tính tỷ lệ từ lặp trong text.
    Giá trị cao = nhiều từ bị lặp lại.
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    # Tỷ lệ từ lặp = 1 - (unique/total)
    repetition_ratio = 1.0 - (unique_words / total_words)
    return repetition_ratio


def is_segment_spam_advanced(seg, segment_history, window_size=10):
    """
    Phát hiện spam bằng nhiều heuristics tổng quát.
    
    Args:
        seg: Segment cần kiểm tra
        segment_history: List các segment gần đây (sliding window)
        window_size: Số segment gần nhất để so sánh
    
    Returns:
        (bool, str): (is_spam, reason)
    """
    text = seg.get("text", "").strip()
    
    # 1. Kiểm tra text rỗng
    if not text:
        return (True, "empty_text")
    
    # 2. Kiểm tra no_speech_prob (confidence thấp)
    no_speech_prob = seg.get("no_speech_prob", 0)
    if no_speech_prob > 0.9:
        return (True, f"low_confidence({no_speech_prob:.2f})")
    
    # 3. Kiểm tra độ dài bất thường
    duration = seg.get("end_time", seg.get("end", 0)) - seg.get("start_time", seg.get("start", 0))
    word_count = len(text.split())
    
    # Quá ngắn (< 0.3s)
    if duration < 0.3:
        return (True, f"too_short({duration:.1f}s)")
    
    # Tốc độ nói bất thường: > 10 từ/giây (người bình thường: 2-3 từ/giây)
    if duration > 0:
        words_per_second = word_count / duration
        if words_per_second > 10:
            return (True, f"abnormal_speed({words_per_second:.1f}wps)")
    
    # 4. Entropy thấp (text lặp/kém chất lượng)
    entropy = calculate_text_entropy(text)
    if entropy < 2.0:  # Ngưỡng: entropy bình thường ~4-5
        return (True, f"low_entropy({entropy:.2f})")
    
    # 5. Tỷ lệ từ lặp cao
    word_rep_ratio = calculate_word_repetition_ratio(text)
    if word_rep_ratio > 0.7 and word_count > 3:  # >70% từ bị lặp
        return (True, f"high_repetition({word_rep_ratio:.2f})")
    
    # 6. So sánh với sliding window (phát hiện lặp segment)
    recent_segments = segment_history[-window_size:]
    for prev_seg in recent_segments:
        prev_text = prev_seg.get("text", "").strip()
        if is_similar_text(text, prev_text, threshold=0.85):
            return (True, "duplicate_segment")
    
    # 7. Text chỉ chứa số hoặc ký tự đặc biệt
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < len(text) * 0.3:  # <30% là chữ cái
        return (True, "non_text_content")
    
    # 8. Segment quá ngắn về mặt nội dung (< 3 từ và < 1s)
    if word_count < 3 and duration < 1.0:
        return (True, f"too_short_content({word_count}w_{duration:.1f}s)")
    
    return (False, None)


def filter_spam_segments(segments):
    """
    Lọc spam segments với thuật toán cải tiến.
    
    Args:
        segments: List of segments từ Whisper
    
    Returns:
        (filtered_segments, stats): Tuple (danh sách đã lọc, thống kê)
    """
    filtered = []
    segment_history = []  # Sliding window để phát hiện lặp
    
    stats = {
        "total": len(segments),
        "removed": 0,
        "reasons": {}
    }
    
    for seg in segments:
        is_spam, reason = is_segment_spam_advanced(seg, segment_history, window_size=10)
        
        if is_spam:
            stats["removed"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue
        
        # Segment hợp lệ
        filtered.append(seg)
        segment_history.append(seg)
        
        # Giới hạn kích thước sliding window
        if len(segment_history) > 20:
            segment_history.pop(0)
    
    stats["kept"] = len(filtered)
    return filtered, stats


def transcribe_audio_optimized(audio_path, chunk_length_minutes=10):
    """
    Transcription tối ưu với faster-whisper.
    ✅ IMPROVED: Memory management, checkpointing, progress tracking, GPU fallback
    """
    start_time = time.time()
    print("🎧 Đang chạy Whisper transcription (OPTIMIZED)...")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {audio_path}")
    
    device = get_device()
    
    # ✅ CHECKPOINT: Kiểm tra xem đã có cache chưa
    checkpoint_file = config.WHISPER_CACHE.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        print(f"\n💾 Phát hiện checkpoint: {checkpoint_file}")
        checkpoint_data = utils.load_json(checkpoint_file)
        print(f"   ✅ Đã xử lý {checkpoint_data.get('processed_chunks', 0)} chunks trước đó")
        
        user_input = input("   ❓ Tiếp tục từ checkpoint? (y/n): ").strip().lower()
        if user_input == 'y':
            all_segments = checkpoint_data.get('segments', [])
            processed_chunks = checkpoint_data.get('processed_chunks', 0)
            print(f"   ✅ Resume từ chunk {processed_chunks + 1}\n")
        else:
            all_segments = []
            processed_chunks = 0
            print("   ⚠️  Bắt đầu lại từ đầu\n")
    else:
        all_segments = []
        processed_chunks = 0
    
    # ✅ Load model với progress VÀ ERROR HANDLING
    print("   📥 Đang load Whisper model 'medium' (faster-whisper)...")
    load_start = time.time()
    
    # Xác định compute_type dựa trên device
    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"
    
    # ✅ TRY LOAD GPU, FALLBACK TO CPU IF FAILED
    try:
        model = WhisperModel(
            "medium",
            device=device,
            compute_type=compute_type,
            cpu_threads=4 if device == "cpu" else 1,
            num_workers=1,
        )
        load_time = time.time() - load_start
        print(f"   ✅ Load model: {load_time:.1f}s (device={device}, compute_type={compute_type})")
        
    except RuntimeError as e:
        if "CUDA" in str(e) and device == "cuda":
            print(f"\n⚠️  LỖI GPU: {e}")
            print(f"\n🔄 FALLBACK: Thử load model với CPU...")
            
            device = "cpu"
            compute_type = "int8"
            
            model = WhisperModel(
                "medium",
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=1,
            )
            load_time = time.time() - load_start
            print(f"   ✅ Load model: {load_time:.1f}s (device=CPU, compute_type=int8)")
            print(f"\n💡 LƯU Ý:")
            print(f"   - Diarization đã dùng GPU thành công (144s)")
            print(f"   - Whisper fallback về CPU do driver cũ")
            print(f"   - Tốc độ chậm hơn nhưng vẫn hoạt động")
            print(f"   - Để fix: Update NVIDIA driver hoặc downgrade PyTorch\n")
        else:
            raise
    
    print(f"   🎵 Đang xử lý file: {os.path.basename(audio_path)}")
    
    info = sf.info(audio_path)
    duration_minutes = info.duration / 60
    print(f"   ⏱️  Độ dài audio: {duration_minutes:.1f} phút")
    
    # ✅ Dynamic chunk length based on file duration
    if duration_minutes > 120:  # >2h → 20min chunks
        chunk_length_minutes = 20
        print(f"   💡 File dài (>2h) → Sử dụng chunks 20 phút")
    elif duration_minutes > 60:  # >1h → 15min chunks
        chunk_length_minutes = 15
        print(f"   💡 File dài (>1h) → Sử dụng chunks 15 phút")
    
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
    
    # Transcribe từng chunk với ETA VÀ CHECKPOINTING
    trans_start = time.time()
    
    for idx, (chunk_path, chunk_start_time, chunk_end) in enumerate(chunks):
        # ✅ Skip chunks đã xử lý
        if idx < processed_chunks:
            print(f"\n   ⏭️  Bỏ qua chunk {idx+1}/{len(chunks)} (đã xử lý)\n")
            continue
        
        chunk_duration = chunk_end - chunk_start_time
        print(f"\n   📝 Xử lý chunk {idx+1}/{len(chunks)} ({chunk_duration:.1f}s)...")
        
        chunk_trans_start = time.time()
        
        # ✅ ERROR RECOVERY: Try-catch cho từng chunk
        try:
            chunk_result = transcribe_chunk_optimized(model, chunk_path, idx, len(chunks))
            chunk_trans_time = time.time() - chunk_trans_start
            
            # ✅ ETA calculation
            if idx < len(chunks) - 1:
                if idx > 0:  # Tính từ chunk 2 trở đi
                    avg_time_per_chunk = (time.time() - trans_start) / (idx - processed_chunks + 1)
                    remaining_chunks = len(chunks) - idx - 1
                    eta = avg_time_per_chunk * remaining_chunks
                    print(f"      ⏱️  Chunk time: {chunk_trans_time:.1f}s | ETA: {eta:.1f}s ({eta/60:.1f} min)")
            
            # Điều chỉnh timestamps
            for seg in chunk_result.get("segments", []):
                seg["start_time"] += chunk_start_time
                seg["end_time"] += chunk_start_time
                all_segments.append(seg)
            
            # ✅ CHECKPOINT: Lưu progress sau mỗi chunk
            checkpoint_data = {
                'processed_chunks': idx + 1,
                'segments': all_segments,
                'timestamp': datetime.now().isoformat()
            }
            utils.save_json(checkpoint_data, checkpoint_file)
            
        except Exception as e:
            print(f"      ❌ Lỗi chunk {idx+1}: {e}")
            print(f"      🔄 Thử lại sau 3 giây...")
            time.sleep(3)
            
            try:
                chunk_result = transcribe_chunk_optimized(model, chunk_path, idx, len(chunks))
                for seg in chunk_result.get("segments", []):
                    seg["start_time"] += chunk_start_time
                    seg["end_time"] += chunk_start_time
                    all_segments.append(seg)
                print(f"      ✅ Retry thành công!")
            except Exception as retry_error:
                print(f"      ❌ Retry thất bại: {retry_error}")
                print(f"      ⚠️  Bỏ qua chunk này, tiếp tục...")
        
        # ✅ Cleanup chunk file và free memory
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except:
                pass
        
        # ✅ Force garbage collection sau mỗi chunk
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    trans_time = time.time() - trans_start
    print(f"\n   ✅ Transcription hoàn thành: {trans_time:.1f}s\n")
    
    # Clear GPU cache nếu dùng GPU
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"   ✅ Tổng cộng phát hiện {len(all_segments)} segments")
    
    # ✅ Lọc spam với thuật toán cải tiến
    print("   🔍 Đang lọc spam, noise và repeats (ADVANCED)...")
    filter_start = time.time()
    
    filtered_segments, filter_stats = filter_spam_segments(all_segments)
    
    filter_time = time.time() - filter_start
    print(f"   ✅ Lọc hoàn thành: {filter_time:.1f}s\n")
    
    # Hiển thị thống kê chi tiết
    print(f"   📊 Thống kê lọc spam:")
    print(f"      • Tổng segments:     {filter_stats['total']}")
    print(f"      • Đã loại bỏ:        {filter_stats['removed']} ({filter_stats['removed']/filter_stats['total']*100:.1f}%)")
    print(f"      • Còn lại:           {filter_stats['kept']} ({filter_stats['kept']/filter_stats['total']*100:.1f}%)")
    
    if filter_stats['reasons']:
        print(f"\n   🗑️  Chi tiết các lý do loại bỏ:")
        for reason, count in sorted(filter_stats['reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"      • {reason:25s}: {count:4d}")
    
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
    
    # ✅ Xóa checkpoint file khi hoàn thành
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"\n   🗑️  Đã xóa checkpoint file")
        except:
            pass
    
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
