"""Combine all results into final output."""
import json, re
import os
import time
from datetime import datetime
from difflib import SequenceMatcher


def is_similar_text(text1, text2, threshold=0.85):
    """Kiểm tra 2 text có giống nhau không."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > threshold


def load_diarization(file_path):
    """Đọc file diarization.txt và chuyển thành list các đoạn."""
    diarization_segments = []
    time_pattern = re.compile(r"([\d.]+)s\s+–\s+([\d.]+)s\s+:\s+(SPEAKER_\d+)")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = time_pattern.search(line)
            if match:
                start, end, speaker = match.groups()
                diarization_segments.append({
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker
                })
    return diarization_segments


def load_whisper_json(file_path):
    """Đọc file Whisper JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = []
    for seg in data.get("segments", []):
        segment_data = {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        }
        # ✅ Không thêm words vào đây nữa
        segments.append(segment_data)
    
    return segments


def assign_speakers(diarization_segments, transcript_segments):
    """Gán speaker cho từng đoạn transcript."""
    results = []
    prev_texts = []
    window_size = 5
    repeat_count = 0
    
    for seg in transcript_segments:
        speaker_label = "UNKNOWN"
        for d in diarization_segments:
            if not (seg["end"] < d["start"] or seg["start"] > d["end"]):
                speaker_label = d["speaker"]
                break
        
        text = seg["text"]
        
        # Phát hiện lặp
        is_repeat = False
        for prev_text in prev_texts[-window_size:]:
            if is_similar_text(text, prev_text, threshold=0.85):
                repeat_count += 1
                is_repeat = True
                print(f"   🗑️  Bỏ qua repeat: {text[:50]}...")
                break
        
        if is_repeat:
            continue
        
        # ✅ FIX: Chỉ lưu 4 trường cơ bản, không có words
        results.append({
            "speaker": speaker_label,
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })
        
        # Update sliding window
        prev_texts.append(text)
        if len(prev_texts) > window_size * 2:
            prev_texts.pop(0)
    
    print(f"   🗑️  Đã loại bỏ {repeat_count} repeats trong combine")
    return results


def combine_results(diarization_path, transcription_path, gender_path, output_path):
    """Kết hợp tất cả kết quả."""
    start_time = time.time()
    start_dt = datetime.now().strftime("%H:%M:%S")
    print(f"🔗 Đang kết hợp kết quả...")
    print(f"   ⏰ Bắt đầu lúc: {start_dt}")
    
    diar_segments = load_diarization(diarization_path)
    transcript_segments = load_whisper_json(transcription_path)

    with open(gender_path, "r", encoding="utf-8") as f:
        gender_info = json.load(f)

    combined = assign_speakers(diar_segments, transcript_segments)
    for seg in combined:
        seg["gender"] = gender_info.get(seg["speaker"], "Unknown")

    # ✅ Lưu file đơn giản (không có words)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    # ✅ OPTIONAL: Lưu file đầy đủ với words (nếu cần)
    detailed_path = output_path.replace('.json', '_detailed.json')
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(combined_with_words, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu kết quả tổng hợp vào {output_path}")
    print(f"   📝 File chi tiết: {detailed_path}")
    
    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    print(f"⏱️  Thời gian combine: {elapsed:.2f}s")
    print(f"   ⏰ Kết thúc lúc: {end_dt}")
    return output_path
