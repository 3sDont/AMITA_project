"""
Data Processor - Xử lý và merge segments
Mục đích: Tập trung logic xử lý segments, loại bỏ duplicate trong các module
"""
from typing import List, Dict, Tuple, Optional
from text_utils import is_similar_text


def merge_adjacent_segments(
    segments: List[Dict],
    gap_threshold: float = 0.5,
    same_speaker_only: bool = True
) -> List[Dict]:
    """
    Gộp các segments liền kề (đã move từ diarization.py).
    
    Args:
        segments: List segments cần merge
        gap_threshold: Khoảng cách tối đa giữa 2 segments để merge (giây)
        same_speaker_only: Chỉ merge segments cùng speaker
    
    Returns:
        List[Dict]: Segments đã được merge
    
    Use case:
        - Sau diarization chunked → merge lại segments bị cắt
        - Sau transcription → merge các đoạn nói liền kề
    """
    if not segments:
        return []
    
    # Sort theo thời gian bắt đầu
    sorted_segs = sorted(segments, key=lambda x: x.get('start_time', x.get('start', 0)))
    
    merged = []
    current = sorted_segs[0].copy()
    
    for seg in sorted_segs[1:]:
        seg_start = seg.get('start_time', seg.get('start', 0))
        seg_end = seg.get('end_time', seg.get('end', 0))
        cur_end = current.get('end_time', current.get('end', 0))
        
        # Kiểm tra điều kiện merge
        should_merge = False
        
        # Điều kiện 1: Gap nhỏ hơn threshold
        if seg_start - cur_end <= gap_threshold:
            # Điều kiện 2: Cùng speaker (nếu required)
            if same_speaker_only:
                cur_speaker = current.get('speaker', None)
                seg_speaker = seg.get('speaker', None)
                should_merge = (cur_speaker == seg_speaker)
            else:
                should_merge = True
        
        if should_merge:
            # Merge: mở rộng end_time
            current['end_time'] = seg_end
            if 'end' in current:
                current['end'] = seg_end
            
            # Merge text nếu có
            if 'text' in current and 'text' in seg:
                current['text'] += " " + seg['text']
            
            # Merge words nếu có
            if 'words' in current and 'words' in seg:
                current['words'].extend(seg.get('words', []))
        else:
            # Không merge → save current và start new
            merged.append(current)
            current = seg.copy()
    
    # Append segment cuối
    merged.append(current)
    
    return merged


def normalize_segment_format(segments: List[Dict]) -> List[Dict]:
    """
    Chuẩn hóa format của segments.
    
    Chuyển đổi:
        - start/end → start_time/end_time (consistent naming)
        - Thêm các fields mặc định nếu thiếu
    
    Args:
        segments: List segments với format không đồng nhất
    
    Returns:
        List[Dict]: Segments với format chuẩn
    """
    normalized = []
    
    for seg in segments:
        new_seg = {}
        
        # Chuẩn hóa time fields
        new_seg['start_time'] = seg.get('start_time', seg.get('start', 0.0))
        new_seg['end_time'] = seg.get('end_time', seg.get('end', 0.0))
        
        # Copy các fields khác
        for key in ['speaker', 'text', 'gender', 'words', 'no_speech_prob']:
            if key in seg:
                new_seg[key] = seg[key]
        
        # Default values
        if 'speaker' not in new_seg:
            new_seg['speaker'] = 'UNKNOWN'
        
        if 'text' not in new_seg:
            new_seg['text'] = ''
        
        normalized.append(new_seg)
    
    return normalized


def get_segments_stats(segments: List[Dict]) -> Dict:
    """
    Tính thống kê về segments.
    
    Returns:
        Dict: Thống kê bao gồm:
            - total_segments: Tổng số segments
            - total_duration: Tổng thời lượng (giây)
            - speakers: Dict {speaker: count}
            - avg_segment_duration: Độ dài trung bình (giây)
            - min_duration: Segment ngắn nhất
            - max_duration: Segment dài nhất
    """
    if not segments:
        return {
            'total_segments': 0,
            'total_duration': 0,
            'speakers': {},
            'avg_segment_duration': 0,
            'min_duration': 0,
            'max_duration': 0
        }
    
    total_duration = 0
    speakers = {}
    durations = []
    
    for seg in segments:
        start = seg.get('start_time', seg.get('start', 0))
        end = seg.get('end_time', seg.get('end', 0))
        duration = end - start
        
        total_duration += duration
        durations.append(duration)
        
        speaker = seg.get('speaker', 'UNKNOWN')
        speakers[speaker] = speakers.get(speaker, 0) + 1
    
    return {
        'total_segments': len(segments),
        'total_duration': total_duration,
        'speakers': speakers,
        'avg_segment_duration': total_duration / len(segments),
        'min_duration': min(durations),
        'max_duration': max(durations)
    }
