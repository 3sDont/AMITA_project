"""Phân tích chất lượng diarization."""
import json
import sys


def analyze_diarization_quality(diar_path, transcription_path):
    """
    Phân tích kết quả diarization và đưa ra khuyến nghị.
    
    Args:
        diar_path: File diarization
        transcription_path: File transcription
    """
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
    
    # Thống kê
    print(f"👥 Tổng số người nói: {len(speakers)}")
    print(f"⏱️  Tổng thời gian: {total_duration:.1f}s\n")
    
    print("📊 Chi tiết:")
    for speaker, stats in sorted(speakers.items()):
        percentage = (stats['total_time'] / total_duration) * 100
        avg_duration = stats['total_time'] / stats['count']
        
        print(f"   {speaker}:")
        print(f"      - Số lần nói: {stats['count']}")
        print(f"      - Tổng thời gian: {stats['total_time']:.1f}s ({percentage:.1f}%)")
        print(f"      - Trung bình/lần: {avg_duration:.1f}s")
    
    # Khuyến nghị
    print("\n💡 KHUYẾN NGHỊ:")
    if len(speakers) == 1:
        print("   ⚠️  Chỉ 1 người nói! Thử --min-speakers 2 --max-speakers 4")
    elif len(speakers) > 10:
        print("   ⚠️  Quá nhiều người! Thử tăng clustering threshold")
    else:
        print("   ✅ Kết quả hợp lý")
    
    # So sánh với transcription
    if transcription_path:
        try:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                trans_data = json.load(f)
                num_segments = len(trans_data.get('segments', []))
            
            print(f"\n📝 So sánh:")
            print(f"   - Transcript segments: {num_segments}")
            print(f"   - Diarization segments: {sum(s['count'] for s in speakers.values())}")
        except:
            pass
    
    return speakers


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_diarization.py <diar_file> [trans_file]")
        sys.exit(1)
    
    diar_file = sys.argv[1]
    trans_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_diarization_quality(diar_file, trans_file)
