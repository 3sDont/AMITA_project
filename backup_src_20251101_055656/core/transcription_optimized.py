"""
Optimized transcription với faster-whisper.
Nhanh gấp 4x, dùng ít RAM hơn 50% so với openai-whisper.
"""
import json
import time
import os
from datetime import datetime
import soundfile as sf
from pydub import AudioSegment
from difflib import SequenceMatcher
from faster_whisper import WhisperModel


def is_similar_text(text1, text2, threshold=0.85):
    """Kiểm tra 2 text có giống nhau không."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > threshold


# ...existing code for load_diarization_segments, split_audio_by_segments, split_audio_simple...


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
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "no_speech_prob": segment.no_speech_prob,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in (segment.words or [])
            ] if segment.words else []
        })
    
    print(f"      ✅ Chunk {chunk_index + 1}: {len(segments_list)} segments")
    return {"segments": segments_list, "language": info.language}


def transcribe_audio_optimized(audio_path, output_path, chunk_length_minutes=10, diarization_path=None):
    """
    Transcription tối ưu với faster-whisper.
    """
    print("🎧 Đang chạy Whisper transcription (OPTIMIZED)...")
    start_time = time.time()
    start_dt = datetime.now().strftime("%H:%M:%S")
    print(f"   ⏰ Bắt đầu lúc: {start_dt}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {audio_path}")
    
    print("   📥 Đang load Whisper model 'medium' (faster-whisper)...")
    
    # ✅ faster-whisper API
    # compute_type: "int8" (fastest), "float16" (balanced), "float32" (best quality)
    model = WhisperModel(
        "medium",
        device="cpu",           # Hoặc "cuda" nếu có GPU
        compute_type="int8",    # Tiết kiệm RAM, nhanh hơn
        cpu_threads=4,          # Số threads CPU
        num_workers=1,          # Workers cho download model
    )
    
    print(f"   🎵 Đang xử lý file: {os.path.basename(audio_path)}")
    
    info = sf.info(audio_path)
    duration_minutes = info.duration / 60
    print(f"   ⏱️  Độ dài audio: {duration_minutes:.1f} phút")
    
    # Chia chunks (tái sử dụng logic cũ)
    if duration_minutes > chunk_length_minutes:
        print(f"\n   📌 File dài ({duration_minutes:.1f} phút) → Chunked processing")
        if diarization_path and os.path.exists(diarization_path):
            from transcribe import split_audio_by_segments, load_diarization_segments
            print("   🎯 Chia chunks theo diarization segments")
            chunks = split_audio_by_segments(audio_path, diarization_path, chunk_length_minutes)
        else:
            from transcribe import split_audio_simple
            print("   ⚠️  Chia chunks theo thời gian")
            chunks = split_audio_simple(audio_path, chunk_length_minutes)
    else:
        chunks = [(audio_path, 0, info.duration)]
    
    # Transcribe từng chunk
    all_segments = []
    for idx, (chunk_path, chunk_start, chunk_end) in enumerate(chunks):
        chunk_result = transcribe_chunk_optimized(model, chunk_path, idx, len(chunks))
        
        # Điều chỉnh timestamps
        for seg in chunk_result.get("segments", []):
            seg["start"] += chunk_start
            seg["end"] += chunk_start
            all_segments.append(seg)
        
        # Cleanup
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except:
                pass
    
    print(f"\n   ✅ Tổng cộng phát hiện {len(all_segments)} segments")
    
    # Lọc spam, noise, repeats (tái sử dụng logic cũ)
    print("   🔍 Đang lọc spam, noise và repeats...")
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
        duration = seg["end"] - seg["start"]
        if duration < 0.3:
            short_count += 1
            continue
        
        filtered_segments.append(seg)
        prev_texts.append(text)
        if len(prev_texts) > window_size * 2:
            prev_texts.pop(0)
    
    result = {
        "text": " ".join(seg.get("text", "") for seg in filtered_segments),
        "segments": filtered_segments,
        "language": "vi"
    }
    
    print(f"   🗑️  Đã loại bỏ: {spam_count} spam, {silence_count} im lặng, {repeat_count} repeats, {short_count} quá ngắn")
    print(f"   📊 Còn lại: {len(filtered_segments)} segments")
    
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
