"""
Text utilities - Tập trung các hàm xử lý text dùng chung
Mục đích: Loại bỏ duplicate logic giữa các module
"""
import re
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Dict, Tuple


# ==================== TEXT SIMILARITY ====================
def is_similar_text(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """
    So sánh độ giống nhau giữa 2 văn bản.
    
    Args:
        text1: Văn bản thứ nhất
        text2: Văn bản thứ hai
        threshold: Ngưỡng tương đồng (0.0 - 1.0)
    
    Returns:
        bool: True nếu độ tương đồng >= threshold
    
    Sử dụng:
        - Phát hiện transcript bị lặp
        - Merge segments giống nhau
    """
    if not text1 or not text2:
        return False
    
    # Sử dụng SequenceMatcher để tính tỷ lệ giống nhau
    ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return ratio > threshold


# ==================== TEXT CLEANING ====================
def clean_text(text: str) -> str:
    """
    Làm sạch văn bản: loại bỏ whitespace thừa, ký tự lặp.
    
    Args:
        text: Văn bản cần làm sạch
    
    Returns:
        str: Văn bản đã được làm sạch
    
    Xử lý:
        - Normalize whitespace (nhiều space → 1 space)
        - Loại bỏ ký tự lặp >3 lần (aaaa → a)
    """
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)
    
    # Loại bỏ ký tự lặp quá nhiều (giữ tối đa 1 lần)
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    
    return text.strip()


# ==================== TEXT QUALITY ANALYSIS ====================
def calculate_text_entropy(text: str) -> float:
    """
    Tính entropy Shannon để đánh giá chất lượng text.
    
    Args:
        text: Văn bản cần phân tích
    
    Returns:
        float: Giá trị entropy (0.0 - ~5.0)
               - Thấp (<2): text lặp/kém chất lượng
               - Cao (>4): text đa dạng/chất lượng tốt
    
    Sử dụng: Phát hiện spam/noise trong transcript
    """
    if not text or len(text) < 2:
        return 0.0
    
    # Đếm tần suất xuất hiện của mỗi ký tự
    char_counts = Counter(text.lower())
    text_len = len(text)
    
    # Tính entropy theo công thức Shannon
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_len
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def calculate_word_repetition_ratio(text: str) -> float:
    """
    Tính tỷ lệ từ bị lặp trong text.
    
    Args:
        text: Văn bản cần phân tích
    
    Returns:
        float: Tỷ lệ lặp (0.0 - 1.0)
               - 0.0: Không có từ nào lặp (unique)
               - 1.0: Tất cả từ đều giống nhau
    
    Công thức: 1 - (số từ unique / tổng số từ)
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    # Tỷ lệ lặp = 1 - (unique/total)
    repetition_ratio = 1.0 - (unique_words / total_words)
    return repetition_ratio


# ==================== SPAM DETECTION ====================
def is_segment_spam(
    segment: Dict,
    history: List[Dict],
    window_size: int = 10,
    **thresholds
) -> Tuple[bool, str]:
    """
    Phát hiện segment spam/noise bằng nhiều heuristics.
    
    Args:
        segment: Segment cần kiểm tra (dict với keys: text, start, end, no_speech_prob)
        history: List các segment gần đây (sliding window)
        window_size: Số segment gần nhất để so sánh duplicate
        **thresholds: Các ngưỡng tùy chỉnh (optional)
    
    Returns:
        Tuple[bool, str]: (is_spam, reason)
            - is_spam: True nếu là spam
            - reason: Lý do phát hiện spam
    
    Kiểm tra:
        1. Text rỗng
        2. Confidence thấp (no_speech_prob)
        3. Độ dài bất thường
        4. Tốc độ nói bất thường
        5. Entropy thấp
        6. Từ lặp nhiều
        7. Duplicate với history
        8. Non-text content
    """
    # Lấy ngưỡng mặc định hoặc custom
    min_duration = thresholds.get('min_duration', 0.3)
    max_speech_prob = thresholds.get('max_speech_prob', 0.1)
    max_words_per_sec = thresholds.get('max_words_per_sec', 10)
    min_entropy = thresholds.get('min_entropy', 2.0)
    max_repetition = thresholds.get('max_repetition', 0.7)
    similarity_threshold = thresholds.get('similarity_threshold', 0.85)
    
    text = segment.get("text", "").strip()
    
    # 1. Kiểm tra text rỗng
    if not text:
        return (True, "empty_text")
    
    # 2. Kiểm tra confidence (no_speech_prob cao = không phải giọng nói)
    no_speech_prob = segment.get("no_speech_prob", 0)
    if no_speech_prob > (1 - max_speech_prob):  # > 0.9
        return (True, f"low_confidence({no_speech_prob:.2f})")
    
    # 3. Tính thời lượng segment
    start = segment.get("start_time", segment.get("start", 0))
    end = segment.get("end_time", segment.get("end", 0))
    duration = end - start
    word_count = len(text.split())
    
    # Segment quá ngắn
    if duration < min_duration:
        return (True, f"too_short({duration:.1f}s)")
    
    # 4. Tốc độ nói bất thường (>10 từ/giây = có vấn đề)
    if duration > 0:
        words_per_second = word_count / duration
        if words_per_second > max_words_per_sec:
            return (True, f"abnormal_speed({words_per_second:.1f}wps)")
    
    # 5. Entropy thấp = text lặp/kém chất lượng
    entropy = calculate_text_entropy(text)
    if entropy < min_entropy:
        return (True, f"low_entropy({entropy:.2f})")
    
    # 6. Tỷ lệ từ lặp cao
    word_rep_ratio = calculate_word_repetition_ratio(text)
    if word_rep_ratio > max_repetition and word_count > 3:
        return (True, f"high_repetition({word_rep_ratio:.2f})")
    
    # 7. So sánh với history để phát hiện duplicate
    recent_segments = history[-window_size:] if history else []
    for prev_seg in recent_segments:
        prev_text = prev_seg.get("text", "").strip()
        if is_similar_text(text, prev_text, threshold=similarity_threshold):
            return (True, "duplicate_segment")
    
    # 8. Text chỉ chứa số/ký tự đặc biệt (<30% chữ cái)
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < len(text) * 0.3:
        return (True, "non_text_content")
    
    # 9. Segment quá ngắn về nội dung
    if word_count < 3 and duration < 1.0:
        return (True, f"too_short_content({word_count}w_{duration:.1f}s)")
    
    return (False, None)


def filter_spam_segments(
    segments: List[Dict],
    window_size: int = 10,
    **thresholds
) -> Tuple[List[Dict], Dict]:
    """
    Lọc spam cho danh sách segments.
    
    Args:
        segments: List of segments
        window_size: Kích thước sliding window
        **thresholds: Custom thresholds
    
    Returns:
        Tuple[List[Dict], Dict]: (filtered_segments, stats)
            - filtered_segments: Danh sách đã lọc
            - stats: Thống kê {total, removed, kept, reasons}
    """
    filtered = []
    history = []  # Sliding window
    
    stats = {
        "total": len(segments),
        "removed": 0,
        "kept": 0,
        "reasons": {}
    }
    
    for seg in segments:
        # Kiểm tra spam
        is_spam, reason = is_segment_spam(seg, history, window_size, **thresholds)
        
        if is_spam:
            # Segment bị loại bỏ
            stats["removed"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            continue
        
        # Segment hợp lệ
        filtered.append(seg)
        history.append(seg)
        
        # Giới hạn kích thước sliding window
        if len(history) > window_size * 2:
            history.pop(0)
    
    stats["kept"] = len(filtered)
    return filtered, stats


# ==================== TEXT FORMATTING ====================
def format_duration(seconds: float) -> str:
    """
    Format thời gian từ seconds sang MM:SS hoặc HH:MM:SS.
    
    Args:
        seconds: Số giây
    
    Returns:
        str: Thời gian đã format
    
    Examples:
        90.5 → "01:30"
        3665.0 → "01:01:05"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Cắt ngắn text nếu quá dài.
    
    Args:
        text: Văn bản cần cắt
        max_length: Độ dài tối đa
        suffix: Hậu tố thêm vào (mặc định: "...")
    
    Returns:
        str: Văn bản đã cắt
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
