"""Speech-to-text transcription using Whisper."""
import whisper
import json
import time
import os
from datetime import datetime
from pydub import AudioSegment
import soundfile as sf
from difflib import SequenceMatcher

# ✅ FIX: Absolute import thay vì relative
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure torch is available for device checks
import torch

from utils.file_utils import prepare_output_paths


def is_similar_text(text1, text2, threshold=0.85):
    """Kiểm tra 2 text có giống nhau không (dùng similarity ratio)."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > threshold


def load_diarization_segments(diarization_path):
    """Load segments từ diarization result."""
    import re
    segments = []
    time_pattern = re.compile(r"([\d.]+)s\s+–\s+([\d.]+)s\s+:\s+(SPEAKER_\d+)")
    
    with open(diarization_path, "r", encoding="utf-8") as f:
        for line in f:
            match = time_pattern.search(line)
            if match:
                start, end, speaker = match.groups()
                segments.append({
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker
                })
    return segments


def split_audio_by_segments(audio_path, diarization_path, chunk_length_minutes=10):
    """
    Chia audio thành chunks nhưng kết thúc tại segment boundaries.
    
    Args:
        audio_path: Đường dẫn file audio
        diarization_path: File diarization để lấy segment boundaries
        chunk_length_minutes: Độ dài mục tiêu mỗi chunk (phút)
    
    Returns:
        List of (chunk_path, start_time, end_time) tuples
    """
    print(f"   ✂️  Chia audio thông minh theo segment boundaries...")
    
    # Load diarization segments
    if not os.path.exists(diarization_path):
        print("   ⚠️  Không tìm thấy file diarization, chia thông thường...")
        return split_audio_simple(audio_path, chunk_length_minutes)
    
    segments = load_diarization_segments(diarization_path)
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
            if seg['start'] >= current_start and seg['end'] <= total_duration:
                # Nếu segment này gần target_end nhất
                if seg['end'] <= target_end + 60:  # Cho phép sai lệch 60s
                    best_seg = seg
                elif seg['start'] > target_end:
                    break  # Đã quá xa
        
        # Xác định điểm cắt
        if best_seg:
            chunk_end = best_seg['end']
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


def transcribe_chunk(model, chunk_path, chunk_index, total_chunks):
    """Transcribe 1 chunk."""
    print(f"\n   📝 Xử lý chunk {chunk_index + 1}/{total_chunks}...")
    
    result = model.transcribe(
        chunk_path,
        language="vi",
        verbose=False,
        
        # Tối ưu
        beam_size=5,
        best_of=5,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.5,
        condition_on_previous_text=True,
        fp16=torch.cuda.is_available(),
        word_timestamps=True  # ✅ THÊM: Bật timestamps cho từng từ (chính xác hơn)
    )
    
    print(f"      ✅ Chunk {chunk_index + 1}: {len(result.get('segments', []))} segments")
    return result


def transcribe_audio(audio_path, output_path, chunk_length_minutes=10, diarization_path=None):
    """
    Chuyển giọng nói thành văn bản bằng Whisper (tối ưu cho file dài).
    
    Args:
        audio_path: Đường dẫn file audio
        output_path: Đường dẫn output JSON
        chunk_length_minutes: Độ dài mục tiêu mỗi chunk (phút)
        diarization_path: File diarization để chia chunks thông minh (optional)
    """
    print("🎧 Đang chạy Whisper transcription...")
    start_time = time.time()
    start_dt = datetime.now().strftime("%H:%M:%S")
    print(f"   ⏰ Bắt đầu lúc: {start_dt}")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {audio_path}")
    
    print("   📥 Đang load Whisper model 'medium'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   ⚙️  Device: {device}")
    
    model = whisper.load_model("medium", device=device)
    
    print(f"   🎵 Đang xử lý file: {os.path.basename(audio_path)}")
    
    # Lấy thông tin file
    info = sf.info(audio_path)
    duration_minutes = info.duration / 60
    print(f"   ⏱️  Độ dài audio: {duration_minutes:.1f} phút")
    
    # ✅ Quyết định có cần chia chunks không
    if duration_minutes > chunk_length_minutes:
        print(f"\n   📌 File dài ({duration_minutes:.1f} phút) → Sử dụng chunked processing")
        
        # ✅ Chia chunks thông minh nếu có diarization
        if diarization_path and os.path.exists(diarization_path):
            print("   🎯 Sử dụng diarization để chia chunks thông minh")
            chunks = split_audio_by_segments(audio_path, diarization_path, chunk_length_minutes)
        else:
            print("   ⚠️  Không có diarization, chia chunks theo thời gian")
            chunks = split_audio_simple(audio_path, chunk_length_minutes)
    else:
        chunks = [(audio_path, 0, info.duration)]
    
    # Transcribe từng chunk
    all_segments = []
    
    for idx, (chunk_path, chunk_start, chunk_end) in enumerate(chunks):
        chunk_result = transcribe_chunk(model, chunk_path, idx, len(chunks))
        
        # ✅ Điều chỉnh timestamps dựa trên chunk_start
        for seg in chunk_result.get("segments", []):
            seg["start"] += chunk_start
            seg["end"] += chunk_start
            all_segments.append(seg)
        
        # Cleanup chunk files (nếu không phải file gốc)
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except:
                pass
    
    # Tổng hợp kết quả
    result = {
        "text": " ".join(seg.get("text", "") for seg in all_segments),
        "segments": all_segments,
        "language": "vi"
    }
    
    print(f"\n   ✅ Tổng cộng phát hiện {len(all_segments)} segments")
    
    # ✅ Lọc spam và noise
    print("   🔍 Đang lọc spam, noise và repeats...")
    filtered_segments = []
    spam_keywords = [
        "đăng ký kênh",
        "ủng hộ kênh",
        "subscribe",
        "like share",
        "theo dõi",
        "bật chuông",
        "anh cứ cầm đi",  # ✅ THÊM: Phát hiện pattern lặp cụ thể
    ]
    
    spam_count = 0
    silence_count = 0
    repeat_count = 0
    
    prev_texts = []  # ✅ THÊM: Track nhiều text trước đó (sliding window)
    window_size = 5  # ✅ Kiểm tra 5 segments gần nhất
    
    for seg in all_segments:
        text = seg.get("text", "").strip()
        no_speech_prob = seg.get("no_speech_prob", 0)
        
        # Bỏ qua rỗng
        if not text:
            continue
        
        # Bỏ qua im lặng
        if no_speech_prob > 0.9:
            silence_count += 1
            continue
        
        # Bỏ qua spam keywords
        is_spam = any(keyword in text.lower() for keyword in spam_keywords)
        if is_spam:
            spam_count += 1
            continue
        
        # ✅ NÂNG CAO: Phát hiện lặp bằng similarity
        is_repeat = False
        for prev_text in prev_texts[-window_size:]:
            # So sánh similarity với các text gần đây
            if is_similar_text(text, prev_text, threshold=0.85):
                is_repeat = True
                repeat_count += 1
                break
        
        if is_repeat:
            continue
        
        # ✅ Bỏ qua segments quá ngắn (có thể là nhiễu)
        duration = seg["end"] - seg["start"]
        if duration < 0.3:  # < 0.3s
            continue
        
        # Thêm vào kết quả
        filtered_segments.append(seg)
        
        # Update sliding window
        prev_texts.append(text)
        if len(prev_texts) > window_size * 2:  # Giữ tối đa 10 texts
            prev_texts.pop(0)
    
    result["segments"] = filtered_segments
    
    print(f"   🗑️  Đã loại bỏ: {spam_count} spam, {silence_count} im lặng, {repeat_count} repeats")
    print(f"   📊 Còn lại: {len(filtered_segments)} segments")
    
    # ✅ CẢNH BÁO nếu kết quả quá ít
    if len(filtered_segments) < 10 and duration_minutes > 10:
        print("\n   ⚠️⚠️⚠️  CẢNH BÁO: Kết quả có vẻ bất thường!")
        print(f"   Chỉ có {len(filtered_segments)} segments cho audio {duration_minutes:.1f} phút")
        print("   Nguyên nhân có thể:")
        print("   - File audio có vấn đề (nhiễu quá nhiều)")
        print("   - Audio chứa toàn watermark/quảng cáo")
        print("\n   💡 Khuyến nghị:")
        print("   1. Kiểm tra file audio gốc bằng cách nghe thử")
        print("   2. Thử giảm threshold: no_speech_threshold=0.4")
        print("   3. Thử chunk nhỏ hơn: chunk_length_minutes=5\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n✅ Đã lưu file transcription vào {output_path}")
    print(f"⏱️  Thời gian transcription: {elapsed:.2f}s ({elapsed/60:.1f} phút)")
    print(f"   ⏰ Kết thúc lúc: {end_dt}")
    
    total_text_length = sum(len(seg.get("text", "")) for seg in filtered_segments)
    print(f"   📝 Tổng số ký tự: {total_text_length:,}")
    if elapsed > 0:
        print(f"   🎯 Tốc độ xử lý: {duration_minutes / (elapsed / 60):.2f}x realtime")
    
    return output_path
