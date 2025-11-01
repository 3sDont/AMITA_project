"""Combine all results into final output."""
import json, re
import os
import time
from datetime import datetime
from difflib import SequenceMatcher

# ✅ No imports needed - self-contained


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
            "text": seg["text"].strip(),
            "words": seg.get("words", [])  # ✅ Lưu words nếu có
        }
        segments.append(segment_data)
    
    return segments


def assign_speakers(diarization_segments, transcript_segments):
    """Gán speaker cho từng đoạn transcript."""
    results = []
    results_with_words = []  # ✅ Thêm list riêng cho version có words
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
        
        # ✅ Version đơn giản (không có words)
        results.append({
            "speaker": speaker_label,
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })
        
        # ✅ Version đầy đủ (có words)
        results_with_words.append({
            "speaker": speaker_label,
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "words": seg.get("words", [])
        })
        
        # Update sliding window
        prev_texts.append(text)
        if len(prev_texts) > window_size * 2:
            prev_texts.pop(0)
    
    print(f"   🗑️  Đã loại bỏ {repeat_count} repeats trong combine")
    return results, results_with_words  # ✅ Return cả 2


def combine_results(diarization_path, transcription_path, gender_path, output_path):
    """Kết hợp tất cả kết quả."""
    start_time = time.time()
    start_dt = datetime.now().strftime("%H:%M:%S")
    print(f"🔗 Đang kết hợp kết quả...")
    print(f"   ⏰ Bắt đầu lúc: {start_dt}\n")
    
    # ✅ Load diarization
    print("   📂 Load diarization...")
    load_diar_start = time.time()
    diar_segments = load_diarization(diarization_path)
    load_diar_time = time.time() - load_diar_start
    print(f"      ✅ {len(diar_segments)} segments ({load_diar_time:.1f}s)")
    
    # ✅ Load transcription
    print("   📂 Load transcription...")
    load_trans_start = time.time()
    transcript_segments = load_whisper_json(transcription_path)
    load_trans_time = time.time() - load_trans_start
    print(f"      ✅ {len(transcript_segments)} segments ({load_trans_time:.1f}s)")

    # ✅ Load gender
    print("   📂 Load gender info...")
    load_gender_start = time.time()
    with open(gender_path, "r", encoding="utf-8") as f:
        gender_info = json.load(f)
    load_gender_time = time.time() - load_gender_start
    print(f"      ✅ {len(gender_info)} speakers ({load_gender_time:.1f}s)\n")

    # ✅ Assign speakers
    print("   🔀 Gán speakers cho transcripts...")
    assign_start = time.time()
    combined, combined_with_words = assign_speakers(diar_segments, transcript_segments)
    assign_time = time.time() - assign_start
    print(f"      ✅ {len(combined)} segments ({assign_time:.1f}s)\n")
    
    # ✅ Thêm gender info
    print("   👤 Thêm gender info...")
    gender_start = time.time()
    for seg in combined:
        seg["gender"] = gender_info.get(seg["speaker"], "Unknown")
    
    for seg in combined_with_words:
        seg["gender"] = gender_info.get(seg["speaker"], "Unknown")
    gender_time = time.time() - gender_start
    print(f"      ✅ Hoàn thành ({gender_time:.1f}s)\n")

    # ✅ Lưu file
    print("   💾 Đang lưu kết quả...")
    save_start = time.time()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    detailed_path = output_path.replace('.json', '_detailed.json')
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(combined_with_words, f, ensure_ascii=False, indent=2)
    
    save_time = time.time() - save_start
    print(f"      ✅ Lưu file ({save_time:.1f}s)")
    
    print(f"\n✅ Đã lưu kết quả tổng hợp vào {output_path}")
    print(f"   📝 File chi tiết (có words): {detailed_path}")
    
    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n⏱️  THỜI GIAN CHI TIẾT:")
    print(f"   - Load diarization:   {load_diar_time:>6.1f}s")
    print(f"   - Load transcription: {load_trans_time:>6.1f}s")
    print(f"   - Load gender:        {load_gender_time:>6.1f}s")
    print(f"   - Assign speakers:    {assign_time:>6.1f}s")
    print(f"   - Add gender info:    {gender_time:>6.1f}s")
    print(f"   - Save results:       {save_time:>6.1f}s")
    print(f"   {'─'*35}")
    print(f"   🕓 TOTAL:             {elapsed:>6.1f}s")
    print(f"   ⏰ Kết thúc lúc: {end_dt}\n")
    
    return output_path
