"""
Combine all results into final output.

✅ REFACTORED: Chỉ tập trung vào combine logic, không làm nhiệm vụ khác
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
import utils

# ✅ IMPORT SHARED UTILITIES
from text_utils import is_similar_text


def assign_speakers(diarization_segments, transcript_segments):
    """
    Gán speaker cho từng đoạn transcript.
    
    ✅ SIMPLIFIED: Chỉ làm assign logic, không filter spam (đã filter ở whisper)
    
    Args:
        diarization_segments: List of diarization segments
        transcript_segments: List of transcript segments
    
    Returns:
        Tuple[List[Dict], List[Dict]]: (simple_output, detailed_output)
            - simple_output: Version đơn giản (không có words)
            - detailed_output: Version đầy đủ (có words)
    """
    results = []
    results_with_words = []
    
    for seg in transcript_segments:
        # Xác định speaker dựa trên overlap thời gian
        speaker_label = "UNKNOWN"
        seg_start = seg.get("start_time", seg.get("start"))
        seg_end = seg.get("end_time", seg.get("end"))
        
        for d in diarization_segments:
            d_start = d.get("start_time", d.get("start"))
            d_end = d.get("end_time", d.get("end"))
            
            # Kiểm tra overlap
            if not (seg_end < d_start or seg_start > d_end):
                speaker_label = d["speaker"]
                break
        
        text = seg.get("text", "").strip()
        
        # Version đơn giản
        results.append({
            "speaker": speaker_label,
            "start": seg_start,
            "end": seg_end,
            "text": text
        })
        
        # Version đầy đủ
        results_with_words.append({
            "speaker": speaker_label,
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "words": seg.get("words", [])
        })
    
    return results, results_with_words


def combine_results():
    """
    ✅ MAIN API: Kết hợp tất cả kết quả.
    
    Changes:
        - Đơn giản hóa: chỉ combine, không làm nhiệm vụ khác
        - Lọc spam đã được xử lý ở whisper.py
        - Format, dialog generation sẽ do LLM module xử lý
    """
    print(f"🔗 Đang kết hợp kết quả...")
    
    # Load dữ liệu
    print("   📂 Đang load dữ liệu...")
    
    diar_data = utils.load_json(config.DIARIZATION_CACHE)
    diar_segments = diar_data if isinstance(diar_data, list) else diar_data.get('segments', [])
    
    whisper_data = utils.load_json(config.WHISPER_CACHE)
    transcript_segments = whisper_data if isinstance(whisper_data, list) else whisper_data.get('segments', [])
    
    gender_info = utils.load_json(config.GENDER_CACHE)
    
    print(f"         - Diarization: {len(diar_segments)} segments")
    print(f"         - Transcripts: {len(transcript_segments)} segments")
    print(f"         - Gender info: {len(gender_info)} speakers\n")

    # Assign speakers
    print("   🔀 Gán speakers cho transcripts...")
    combined, combined_with_words = assign_speakers(diar_segments, transcript_segments)
    
    # Thêm gender info
    print("   👤 Thêm gender info...")
    for seg in combined:
        seg["gender"] = gender_info.get(seg["speaker"], "Unknown")
    
    for seg in combined_with_words:
        seg["gender"] = gender_info.get(seg["speaker"], "Unknown")

    # Lưu file
    print("   💾 Đang lưu kết quả...")
    
    utils.save_json(combined, config.COMBINING_CACHE)
    utils.save_json(combined_with_words, config.COMBINING_DETAILED_CACHE)
    
    print(f"\n✅ Đã lưu kết quả tổng hợp vào {config.COMBINING_CACHE}")
    print(f"   📝 File chi tiết (có words): {config.COMBINING_DETAILED_CACHE}")

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
