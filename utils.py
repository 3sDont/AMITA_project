"""
Hàm tiện ích dùng chung cho toàn bộ pipeline
"""

import os
import json
import datetime
from typing import List, Dict, Any


def save_json(data: Any, filepath: str, indent: int = 2):
    """Lưu dữ liệu ra file JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"✓ Đã lưu: {filepath}")


def load_json(filepath: str) -> Any:
    """Đọc dữ liệu từ file JSON"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Đã đọc: {filepath}")
    return data


def check_file_exists(filepath: str) -> bool:
    """Kiểm tra file có tồn tại không"""
    return os.path.exists(filepath)


def get_file_size(filepath: str) -> str:
    """Lấy kích thước file dạng string"""
    if not os.path.exists(filepath):
        return "0 bytes"
    
    size = os.path.getsize(filepath)
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def get_current_time() -> str:
    """Lấy thời gian hiện tại dạng ISO format"""
    return datetime.datetime.now().isoformat()


def print_section(title: str, width: int = 70):
    """In tiêu đề section"""
    print("\n" + "="*width)
    print(title)
    print("="*width)


def print_progress(current: int, total: int, prefix: str = ""):
    """In tiến độ"""
    percentage = (current / total * 100) if total > 0 else 0
    print(f"{prefix}[{current}/{total}] ({percentage:.1f}%)")


def format_time(seconds: float) -> str:
    """Format thời gian từ giây sang MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def calculate_statistics(segments: List[Dict]) -> Dict[str, Any]:
    """Tính thống kê từ segments"""
    if not segments:
        return {
            "total_segments": 0,
            "total_speakers": 0,
            "total_duration": 0,
            "total_words": 0,
            "total_chars": 0,
            "speakers": {}
        }
    
    speakers = list(set(s['speaker'] for s in segments))
    total_duration = sum(s['end_time'] - s['start_time'] for s in segments)
    total_words = sum(len(s['text'].split()) for s in segments)
    total_chars = sum(len(s['text']) for s in segments)
    
    # Thống kê theo speaker
    speaker_stats = {}
    for speaker in speakers:
        speaker_segments = [s for s in segments if s['speaker'] == speaker]
        speaker_duration = sum(s['end_time'] - s['start_time'] for s in speaker_segments)
        speaker_words = sum(len(s['text'].split()) for s in speaker_segments)
        percentage = (speaker_duration / total_duration * 100) if total_duration > 0 else 0
        
        speaker_stats[speaker] = {
            "segments": len(speaker_segments),
            "duration": speaker_duration,
            "percentage": percentage,
            "words": speaker_words
        }
    
    return {
        "total_segments": len(segments),
        "total_speakers": len(speakers),
        "total_duration": total_duration,
        "total_words": total_words,
        "total_chars": total_chars,
        "speakers": speaker_stats
    }


def print_statistics(stats: Dict[str, Any]):
    """In thống kê"""
    print_section("📊 THỐNG KÊ")
    
    print(f"Tổng số phân đoạn: {stats['total_segments']}")
    print(f"Số người nói: {stats['total_speakers']}")
    print(f"Tổng thời lượng: {format_time(stats['total_duration'])} ({stats['total_duration']:.1f}s)")
    print(f"Tổng số từ: {stats['total_words']}")
    print(f"Tổng số ký tự: {stats['total_chars']}")
    
    if stats['speakers']:
        print("\n📊 Thống kê theo người nói:")
        for speaker, data in sorted(stats['speakers'].items()):
            print(f"\n  {speaker}:")
            print(f"    - Số lần nói: {data['segments']}")
            print(f"    - Thời lượng: {format_time(data['duration'])} ({data['percentage']:.1f}%)")
            print(f"    - Số từ: {data['words']}")
