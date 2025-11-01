"""Diarization quality analysis."""
import json

def analyze_diarization_quality(diar_path, transcription_path):
    """Phân tích chất lượng diarization và đưa ra khuyến nghị."""
    print("\n📊 PHÂN TÍCH KẾT QUẢ DIARIZATION\n")
    
    # Đọc diarization
    speakers = {}
    total_duration = 0
    with open(diar_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'SPEAKER_' in line:
                parts = line.strip().split(':')
                times = parts[0].strip()
                speaker = parts[1].strip()
                
                start, end = times.replace('s', '').split('–')
                duration = float(end) - float(start)
                
                if speaker not in speakers:
                    speakers[speaker] = {'count': 0, 'total_time': 0}
                speakers[speaker]['count'] += 1
                speakers[speaker]['total_time'] += duration
                total_duration += duration
    
    # Hiển thị thống kê
    print(f"👥 Tổng số người nói: {len(speakers)}")
    print(f"⏱️  Tổng thời gian: {total_duration:.1f}s\n")
    
    print("📊 Thống kê chi tiết:")
    for speaker, stats in sorted(speakers.items()):
        percentage = (stats['total_time'] / total_duration) * 100
        print(f"   {speaker}:")
        print(f"      - Số lần nói: {stats['count']}")
        print(f"      - Tổng thời gian: {stats['total_time']:.1f}s ({percentage:.1f}%)")
        print(f"      - Trung bình/lần: {stats['total_time']/stats['count']:.1f}s")
    
    # Đưa ra khuyến nghị
    print("\n💡 KHUYẾN NGHỊ:")
    if len(speakers) == 1:
        print("   ⚠️  Chỉ phát hiện 1 người nói!")
        print("   → Thử lại với --min-speakers 2 --max-speakers 4")
        print("   → Kiểm tra audio có thực sự có nhiều người không")
    elif len(speakers) > 10:
        print("   ⚠️  Phát hiện quá nhiều người nói (có thể nhầm)!")
        print("   → Thử tăng clustering threshold")
        print("   → Hoặc chỉ định --max-speakers <số_thực_tế>")
    else:
        print("   ✅ Kết quả hợp lý!")
    
    # So sánh với transcription
    with open(transcription_path, 'r', encoding='utf-8') as f:
        trans_data = json.load(f)
        num_segments = len(trans_data.get('segments', []))
    
    print(f"\n📝 So sánh:")
    print(f"   - Số segments transcript: {num_segments}")
    print(f"   - Số segments diarization: {sum(s['count'] for s in speakers.values())}")
    
    return speakers

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analyze_diarization.py <diar_file> <trans_file>")
        sys.exit(1)
    
    analyze_diarization_quality(sys.argv[1], sys.argv[2])
