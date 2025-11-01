"""
Parallel transcription với multiprocessing.
Tăng tốc 2-4x trên CPU nhiều cores.
"""
import json
import time
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from faster_whisper import WhisperModel
from transcribe import split_audio_by_segments, split_audio_simple
import soundfile as sf

def transcribe_chunk_worker(args):
    """Worker function cho multiprocessing."""
    chunk_path, chunk_index, total_chunks, chunk_start = args
    
    # Load model trong mỗi worker
    model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=2)
    
    segments, info = model.transcribe(
        chunk_path,
        language="vi",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )
    
    segments_list = []
    for seg in segments:
        segments_list.append({
            "start": seg.start + chunk_start,  # Điều chỉnh timestamps
            "end": seg.end + chunk_start,
            "text": seg.text,
            "no_speech_prob": seg.no_speech_prob,
        })
    
    print(f"   ✅ Chunk {chunk_index + 1}/{total_chunks}: {len(segments_list)} segments")
    return segments_list


def transcribe_audio_parallel(audio_path, output_path, chunk_length_minutes=5, diarization_path=None, num_workers=None):
    """
    Transcribe với multiprocessing (parallel).
    
    Args:
        num_workers: Số workers (mặc định = số CPU cores - 1)
    """
    print("🎧 Đang chạy Whisper transcription (PARALLEL)...")
    start_time = time.time()
    
    info = sf.info(audio_path)
    duration_minutes = info.duration / 60
    
    if duration_minutes > chunk_length_minutes:
        if diarization_path and os.path.exists(diarization_path):
            chunks = split_audio_by_segments(audio_path, diarization_path, chunk_length_minutes)
        else:
            chunks = split_audio_simple(audio_path, chunk_length_minutes)
    else:
        chunks = [(audio_path, 0, info.duration)]
    
    # Chuẩn bị args cho workers
    worker_args = [
        (chunk_path, idx, len(chunks), chunk_start)
        for idx, (chunk_path, chunk_start, _) in enumerate(chunks)
    ]
    
    # Parallel processing
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"   🔀 Sử dụng {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = pool.map(transcribe_chunk_worker, worker_args)
    
    # Gộp kết quả
    all_segments = []
    for segments_list in results:
        all_segments.extend(segments_list)
    
    # Cleanup chunks
    for chunk_path, _, _ in chunks:
        if chunk_path != audio_path:
            try:
                os.remove(chunk_path)
            except:
                pass
    
    result = {"text": " ".join(s["text"] for s in all_segments), "segments": all_segments}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Thời gian transcription (parallel): {elapsed:.2f}s")
    print(f"   🎯 Tốc độ xử lý: {duration_minutes / (elapsed / 60):.2f}x realtime")
    
    return output_path
