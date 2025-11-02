"""Combine all results into final output."""
import sys
import os
# Thêm thư mục gốc vào sys.path để import config và utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
import config
import utils


def is_similar_text(text1, text2, threshold=0.85):
    """Kiểm tra 2 text có giống nhau không."""
    if not text1 or not text2:
        return False
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > threshold


def assign_speakers(diarization_segments, transcript_segments):
    """Gán speaker cho từng đoạn transcript."""
    results = []
    results_with_words = []  # ✅ Thêm list riêng cho version có words
    prev_texts = []
    window_size = 5
    repeat_count = 0
    
    for seg in transcript_segments:
        speaker_label = "UNKNOWN"
        # ✅ So sánh với start_time/end_time từ diarization
        seg_start = seg.get("start_time", seg.get("start"))
        seg_end = seg.get("end_time", seg.get("end"))
        
        for d in diarization_segments:
            d_start = d.get("start_time", d.get("start"))
            d_end = d.get("end_time", d.get("end"))
            # Kiểm tra overlap
            if not (seg_end < d_start or seg_start > d_end):
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
            "start": seg_start,
            "end": seg_end,
            "text": text
        })
        
        # ✅ Version đầy đủ (có words)
        results_with_words.append({
            "speaker": speaker_label,
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "words": seg.get("words", [])
        })
        
        # Update sliding window
        prev_texts.append(text)
        if len(prev_texts) > window_size * 2:
            prev_texts.pop(0)
    
    print(f"   🗑️  Đã loại bỏ {repeat_count} repeats trong combine")
    return results, results_with_words  # ✅ Return cả 2


def combine_results():
    """Kết hợp tất cả kết quả."""
    #start_time = time.time()
    #start_dt = datetime.now().strftime("%H:%M:%S")
    print(f"🔗 Đang kết hợp kết quả...")
    #print(f"   ⏰ Bắt đầu lúc: {start_dt}\n")
    
    # ✅ Load dữ liệu từ các file cache
    print("   📂 Đang load dữ liệu...")
    #load_start = time.time()
    
    # Load diarization
    diar_data = utils.load_json(config.DIARIZATION_CACHE)
    if isinstance(diar_data, list):
        diar_segments = diar_data
    else:
        diar_segments = diar_data.get('segments', [])
    
    # Load whisper transcripts
    whisper_data = utils.load_json(config.WHISPER_CACHE)
    if isinstance(whisper_data, list):
        transcript_segments = whisper_data
    else:
        transcript_segments = whisper_data.get('segments', [])
    
    # Load gender info
    gender_info = utils.load_json(config.GENDER_CACHE)
    
    #load_time = time.time() - load_start
    #print(f"      ✅ Load hoàn thành ({load_time:.1f}s)")
    print(f"         - Diarization: {len(diar_segments)} segments")
    print(f"         - Transcripts: {len(transcript_segments)} segments")
    print(f"         - Gender info: {len(gender_info)} speakers\n")

    # ✅ Assign speakers
    print("   🔀 Gán speakers cho transcripts...")
    #assign_start = time.time()
    combined, combined_with_words = assign_speakers(diar_segments, transcript_segments)
    #assign_time = time.time() - assign_start
    #print(f"      ✅ {len(combined)} segments ({assign_time:.1f}s)\n")
    
    # ✅ Thêm gender info
    print("   👤 Thêm gender info...")
    #gender_start = time.time()
    for seg in combined:
        seg["gender"] = gender_info.get(seg["speaker"], "Unknown")
    
    for seg in combined_with_words:
        seg["gender"] = gender_info.get(seg["speaker"], "Unknown")
    #gender_time = time.time() - gender_start
    #   print(f"      ✅ Hoàn thành ({gender_time:.1f}s)\n")

    # ✅ Lưu file
    print("   💾 Đang lưu kết quả...")
    #save_start = time.time()
    
    utils.save_json(combined, config.COMBINING_CACHE)
    
    utils.save_json(combined_with_words, config.COMBINING_DETAILED_CACHE)
    
    #save_time = time.time() - save_start
    #print(f"      ✅ Lưu file ({save_time:.1f}s)")
    print(f"\n✅ Đã lưu kết quả tổng hợp vào {config.COMBINING_CACHE}")
    print(f"   📝 File chi tiết (có words): {config.COMBINING_DETAILED_CACHE}")

    #elapsed = time.time() - start_time
    #end_dt = datetime.now().strftime("%H:%M:%S")
    
    #print(f"\n⏱️  THỜI GIAN CHI TIẾT:")
    #print(f"   - Load data:          {load_time:>6.1f}s")
    #print(f"   - Assign speakers:    {assign_time:>6.1f}s")
    #print(f"   - Add gender info:    {gender_time:>6.1f}s")
    #print(f"   - Save results:       {save_time:>6.1f}s")
    #print(f"   {'─'*35}")
    #print(f"   🕓 TOTAL:             {elapsed:>6.1f}s")
    #print(f"   ⏰ Kết thúc lúc: {end_dt}\n")
    
    return combined


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🔗 GIAI ĐOẠN 3: COMBINING RESULTS")
        print("="*80 + "\n")
        
        # Kiểm tra các file cần thiết
        required_files = {
            "Diarization": config.DIARIZATION_CACHE,
            "Whisper": config.WHISPER_CACHE,
            "Gender": config.GENDER_CACHE
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append((name, path))
        
        if missing_files:
            print("❌ Lỗi: Thiếu các file sau:")
            for name, path in missing_files:
                print(f"   - {name}: {path}")
            print("\n💡 Vui lòng chạy các giai đoạn trước:")
            print("   1. python src/diarization.py")
            print("   2. python src/whisper.py")
            print("   3. python src/gender_classifier.py")
            sys.exit(1)
        
        # Tạo thư mục outputs nếu chưa có
        os.makedirs("outputs", exist_ok=True)
        
        print(f"📁 Input Files:")
        print(f"   - Diarization: {config.DIARIZATION_CACHE}")
        print(f"   - Whisper:     {config.WHISPER_CACHE}")
        print(f"   - Gender:      {config.GENDER_CACHE}\n")
        
        # Chạy combining
        result = combine_results()
        
        print(f"✓ Kết quả đã lưu tại:")
        print(f"   - Simple:   {config.COMBINING_CACHE}")
        print(f"   - Detailed: {config.COMBINING_DETAILED_CACHE}")
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH GIAI ĐOẠN 3")
        print("="*80)
        print(f"📊 Tổng số segments: {len(result)}")
        
        # Thống kê speakers
        speakers = {}
        for seg in result:
            speaker = seg['speaker']
            speakers[speaker] = speakers.get(speaker, 0) + 1
        
        print(f"👥 Số người nói: {len(speakers)}")
        
        # Preview một số segments
        print(f"\n📋 Preview 3 segments đầu tiên:")
        print("-" * 80)
        for i, seg in enumerate(result[:3]):
            gender_emoji = "👨" if seg.get('gender') == "Male" else "👩" if seg.get('gender') == "Female" else "❓"
            print(f"\n[{i+1}] {seg['start']:.2f}s → {seg['end']:.2f}s")
            print(f"    {gender_emoji} Speaker: {seg['speaker']} ({seg.get('gender', 'Unknown')})")
            print(f"    Text: {seg['text'][:100]}{'...' if len(seg['text']) > 100 else ''}")
        
        if len(result) > 3:
            print(f"\n... và {len(result) - 3} segments khác")
        
        # Thống kê chi tiết
        print("\n" + "="*80)
        print("📊 THỐNG KÊ CHI TIẾT:")
        print("-" * 80)
        for speaker in sorted(speakers.keys()):
            count = speakers[speaker]
            # Tìm gender từ result
            gender = "Unknown"
            for seg in result:
                if seg['speaker'] == speaker:
                    gender = seg.get('gender', 'Unknown')
                    break
            gender_emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
            print(f"   {gender_emoji} {speaker} ({gender}): {count} segments")
        
        print("\n" + "="*80)
        print("💡 TIP: Chạy tiếp giai đoạn 4 (LLM analysis) nếu muốn:")
        print("   python src/step4_llm_analysis.py")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
