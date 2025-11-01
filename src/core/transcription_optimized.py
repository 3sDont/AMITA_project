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
    
    # ✅ Load model với progress
    print("   📥 Đang load Whisper model 'medium' (faster-whisper)...")
    load_start = time.time()
    
    model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )
    
    load_time = time.time() - load_start
    print(f"   ✅ Load model: {load_time:.1f}s\n")
    
    print(f"   🎵 Đang xử lý file: {os.path.basename(audio_path)}")
    
    info = sf.info(audio_path)
    duration_minutes = info.duration / 60
    print(f"   ⏱️  Độ dài audio: {duration_minutes:.1f} phút")
    
    # ✅ Ước tính thời gian
    estimated_time = duration_minutes * 0.6  # Faster-whisper: ~36s/phút
    print(f"   ⏳ Ước tính transcription: ~{estimated_time:.1f} phút\n")
    
    # ✅ Chia chunks với progress
    chunk_start = time.time()
    if duration_minutes > chunk_length_minutes:
        print(f"   📌 File dài ({duration_minutes:.1f} phút) → Chunked processing")
        if diarization_path and os.path.exists(diarization_path):
            from core.transcription import split_audio_by_segments, load_diarization_segments
            print("   🎯 Chia chunks theo diarization segments...")
            chunks = split_audio_by_segments(audio_path, diarization_path, chunk_length_minutes)
        else:
            from core.transcription import split_audio_simple
            print("   ⚠️  Chia chunks theo thời gian...")
            chunks = split_audio_simple(audio_path, chunk_length_minutes)
    else:
        chunks = [(audio_path, 0, info.duration)]
    
    chunk_time = time.time() - chunk_start
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
            seg["start"] += chunk_start_time
            seg["end"] += chunk_start_time
            all_segments.append(seg)
        
        # Cleanup
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except:
                pass
    
    trans_time = time.time() - trans_start
    print(f"\n   ✅ Transcription hoàn thành: {trans_time:.1f}s\n")
    
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
        duration = seg["end"] - seg["start"]
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
    
    # ✅ Lưu kết quả với progress
    print("\n   💾 Đang lưu kết quả...")
    save_start = time.time()
    
    result = {
        "text": " ".join(seg.get("text", "") for seg in filtered_segments),
        "segments": filtered_segments,
        "language": "vi"
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    save_time = time.time() - save_start
    print(f"   ✅ Lưu file: {save_time:.1f}s")
    
    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n✅ Đã lưu file transcription vào {output_path}")
    print(f"\n⏱️  THỜI GIAN CHI TIẾT:")
    print(f"   - Load model:       {load_time:>6.1f}s")
    print(f"   - Chia chunks:      {chunk_time:>6.1f}s")
    print(f"   - Transcription:    {trans_time:>6.1f}s")
    print(f"   - Filter:           {filter_time:>6.1f}s")
    print(f"   - Save results:     {save_time:>6.1f}s")
    print(f"   {'─'*35}")
    print(f"   🕓 TOTAL:           {elapsed:>6.1f}s ({elapsed/60:.1f} phút)")
    print(f"   ⏰ Kết thúc lúc: {end_dt}")
    
    total_text_length = sum(len(seg.get("text", "")) for seg in filtered_segments)
    print(f"\n   📝 Tổng số ký tự: {total_text_length:,}")
    if elapsed > 0:
        speedup = duration_minutes / (elapsed / 60)
        print(f"   🎯 Tốc độ xử lý: {speedup:.2f}x realtime")
    
    return output_path
