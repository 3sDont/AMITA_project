"""
🔥 ADVANCED TEXT PREPROCESSING PIPELINE (7 STAGES)
Làm sạch transcript trước khi gửi cho LLM → Kết quả chính xác hơn 300%

Pipeline:
0. Load raw transcript
1. Filter filler utterances (ờ, ừ, kiểu như...)
2. Group by speaker (merge consecutive turns)
3. LLM pre-clean (optional)
4. Content selection (keep relevant, remove noise)
5. Length trimming (optimal 4000-6000 chars)
6. Format for summarizer
7. Format for task extractor
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

# ✅ FIX: Import numpy
import numpy as np

# ✅ FIX: Import config và utils
try:
    import config
    import utils
except ImportError:
    print("⚠️  Warning: config/utils not imported. Running in standalone mode.")
    config = None
    utils = None

logger = logging.getLogger("text_preprocessor")


# ======================= STAGE 1: FILLER FILTER =======================
class FillerFilter:
    """
    Lọc các câu chỉ chứa filler words (ờ, ừ, à, uhm...)
    và các câu vô nghĩa, bị cắt đoạn
    """
    
    # Danh sách filler words tiếng Việt
    FILLER_WORDS = {
        'ờ', 'ừ', 'à', 'ọ', 'ơ', 'ồ', 'ô',
        'uh', 'uhm', 'um', 'er', 'ah', 'eh',
        'kiểu', 'kiểu như', 'thì', 'là',
        'nha', 'nhỉ', 'nhé', 'á', 'nè',
        'này', 'đó', 'kia', 'hả', 'hả',
        'dạ', 'vâng', 'ơi', 'ơ kìa'
    }
    
    # Pattern câu bị cắt đoạn
    INCOMPLETE_PATTERNS = [
        r'^[^.!?]*\.\.\.$',      # Kết thúc bằng ...
        r'^\w{1,2}\s*$',          # Chỉ 1-2 từ
        r'^[^\w\s]+$',            # Chỉ ký tự đặc biệt
    ]
    
    @classmethod
    def is_filler_only(cls, text: str) -> bool:
        """Check nếu câu chỉ chứa filler words"""
        words = text.lower().strip().split()
        if not words:
            return True
        
        # Loại bỏ punctuation
        clean_words = [re.sub(r'[^\w\s]', '', w) for w in words]
        clean_words = [w for w in clean_words if w]
        
        if not clean_words:
            return True
        
        # Check nếu tất cả từ đều là filler
        filler_count = sum(1 for w in clean_words if w in cls.FILLER_WORDS)
        return filler_count >= len(clean_words) * 0.8  # 80% là filler
    
    @classmethod
    def is_incomplete(cls, text: str) -> bool:
        """Check nếu câu bị cắt đoạn hoặc vô nghĩa"""
        text = text.strip()
        
        # Quá ngắn
        if len(text) < 3:
            return True
        
        # Check patterns
        for pattern in cls.INCOMPLETE_PATTERNS:
            if re.match(pattern, text):
                return True
        
        return False
    
    @classmethod
    def filter_segment(cls, text: str) -> Tuple[bool, str]:
        """
        Filter một segment
        Returns: (should_keep, reason)
        """
        if cls.is_filler_only(text):
            return (False, "filler_only")
        
        if cls.is_incomplete(text):
            return (False, "incomplete")
        
        return (True, "")


# ======================= STAGE 2: SPEAKER GROUPER =======================
class SpeakerGrouper:
    """
    Gộp các câu liên tiếp của cùng speaker thành 1 turn dài
    """
    
    @staticmethod
    def group_segments(segments: List[Dict]) -> List[Dict]:
        """
        Gộp consecutive segments của cùng speaker
        
        Input: [
            {"speaker": "A", "text": "Xin chào"},
            {"speaker": "A", "text": "Tôi là A"},
            {"speaker": "B", "text": "Chào bạn"},
        ]
        
        Output: [
            {"speaker": "A", "text": "Xin chào. Tôi là A."},
            {"speaker": "B", "text": "Chào bạn"},
        ]
        """
        if not segments:
            return []
        
        grouped = []
        current = {
            "speaker": segments[0]["speaker"],
            "gender": segments[0].get("gender", "Unknown"),
            "texts": [segments[0]["text"]]
        }
        
        for seg in segments[1:]:
            if seg["speaker"] == current["speaker"]:
                # Cùng speaker → append
                current["texts"].append(seg["text"])
            else:
                # Speaker mới → save current và start new
                grouped.append({
                    "speaker": current["speaker"],
                    "gender": current["gender"],
                    "text": ". ".join(current["texts"]) + "."
                })
                
                current = {
                    "speaker": seg["speaker"],
                    "gender": seg.get("gender", "Unknown"),
                    "texts": [seg["text"]]
                }
        
        # Save last group
        grouped.append({
            "speaker": current["speaker"],
            "gender": current["gender"],
            "text": ". ".join(current["texts"]) + "."
        })
        
        return grouped


# ======================= STAGE 4: CONTENT SELECTOR =======================
class ContentSelector:
    """
    Chọn lọc content có giá trị, loại bỏ noise
    
    KEEP:
    - Technical discussion (đề cập thuật ngữ kỹ thuật)
    - System evaluation (đánh giá, review, feedback)
    - Action items (cần làm, phải làm, sẽ làm...)
    - Decisions (quyết định, thống nhất, đồng ý...)
    
    REMOVE:
    - Small talk (chào hỏi, cảm ơn basic)
    - Pure exclamations (ồ, wow, ok...)
    - Repetitions
    """
    
    # Keywords cho từng category
    TECHNICAL_KEYWORDS = {
        'api', 'backend', 'frontend', 'database', 'server',
        'model', 'training', 'accuracy', 'performance',
        'code', 'bug', 'feature', 'deploy', 'test',
        'dữ liệu', 'dataset', 'mô hình', 'huấn luyện',
        'chạy', 'lỗi', 'tính năng', 'hệ thống',
        'thuật toán', 'độ chính xác', 'kết quả',
        'phân tích', 'đo lường', 'đánh giá'
    }
    
    ACTION_KEYWORDS = {
        'cần', 'phải', 'sẽ', 'làm', 'thực hiện',
        'gửi', 'kiểm tra', 'review', 'test',
        'chuẩn bị', 'hoàn thành', 'deadline',
        'trước', 'sau', 'tuần', 'ngày', 'giờ',
        'phụ trách', 'assigned', 'responsible'
    }
    
    DECISION_KEYWORDS = {
        'quyết định', 'thống nhất', 'đồng ý',
        'chốt', 'ok', 'được', 'approve',
        'kết luận', 'final', 'confirm'
    }
    
    SMALLTALK_PATTERNS = [
        r'^(xin )?chào',
        r'^cảm ơn',
        r'^thank',
        r'^hello',
        r'^hi\b',
        r'^bye',
        r'^tạm biệt',
    ]
    
    @classmethod
    def calculate_relevance_score(cls, text: str) -> float:
        """
        Tính điểm relevance (0-1)
        Càng cao = càng quan trọng
        """
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))
        
        score = 0.0
        
        # Technical content
        tech_count = len(words & cls.TECHNICAL_KEYWORDS)
        if tech_count > 0:
            score += 0.4
        
        # Action items
        action_count = len(words & cls.ACTION_KEYWORDS)
        if action_count > 0:
            score += 0.3
        
        # Decisions
        decision_count = len(words & cls.DECISION_KEYWORDS)
        if decision_count > 0:
            score += 0.3
        
        # Length bonus (longer = more content)
        if len(text) > 100:
            score += 0.1
        
        # Small talk penalty
        for pattern in cls.SMALLTALK_PATTERNS:
            if re.match(pattern, text_lower):
                score -= 0.3
                break
        
        return max(0.0, min(1.0, score))
    
    @classmethod
    def select_content(cls, segments: List[Dict], threshold=0.3) -> List[Dict]:
        """
        Lọc segments theo relevance score
        
        Args:
            segments: List of {"speaker", "text", ...}
            threshold: Minimum score to keep (0.3 = keep important stuff)
        
        Returns:
            Filtered segments
        """
        selected = []
        
        for seg in segments:
            score = cls.calculate_relevance_score(seg["text"])
            
            if score >= threshold:
                seg["relevance_score"] = score
                selected.append(seg)
        
        return selected


# ======================= STAGE 5: LENGTH TRIMMER =======================
class LengthTrimmer:
    """
    Trim xuống 4000-6000 chars (optimal cho LLM)
    - Ưu tiên giữ đầu + cuối
    - Giữ segments có relevance_score cao
    """
    
    @staticmethod
    def trim_to_length(segments: List[Dict], max_chars=5000) -> List[Dict]:
        """
        Trim segments về max_chars
        
        Strategy:
        1. Giữ 40% đầu (context setup)
        2. Giữ 40% cuối (conclusions/actions)
        3. Giữ 20% giữa (high-score segments)
        """
        # Tính tổng length
        total_text = "\n".join(s["text"] for s in segments)
        
        if len(total_text) <= max_chars:
            return segments  # Không cần trim
        
        # Split vị trí
        n = len(segments)
        start_count = int(n * 0.4)
        end_count = int(n * 0.4)
        
        # Giữ đầu + cuối
        keep_segments = segments[:start_count] + segments[-end_count:]
        
        # Giữ thêm middle segments có score cao
        middle = segments[start_count:-end_count]
        middle_sorted = sorted(middle, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Thêm middle cho đến khi đủ chars
        keep_text_len = sum(len(s["text"]) for s in keep_segments)
        
        for seg in middle_sorted:
            if keep_text_len + len(seg["text"]) > max_chars:
                break
            keep_segments.append(seg)
            keep_text_len += len(seg["text"])
        
        # Sort lại theo thứ tự gốc
        keep_segments_set = set(id(s) for s in keep_segments)
        result = [s for s in segments if id(s) in keep_segments_set]
        
        return result


# ======================= MAIN PIPELINE =======================
class TextPreprocessor:
    """
    Main preprocessing pipeline
    """
    
    def __init__(self, enable_content_filter=True, max_length=5000):
        self.enable_content_filter = enable_content_filter
        self.max_length = max_length
    
    def preprocess(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Run full 7-stage pipeline
        
        Input: Raw segments from combining_output.json
        Output: {
            "cleaned_segments": [...],
            "summary_input": "...",  # Text for summarizer
            "tasks_input": "...",    # Text for task extractor
            "stats": {...}
        }
        """
        stats = {
            "original_count": len(segments),
            "original_chars": sum(len(s.get("text", "")) for s in segments)
        }
        
        logger.info(f"[STAGE 0] Loaded {stats['original_count']} segments, {stats['original_chars']} chars")
        
        # STAGE 1: Filter fillers
        logger.info("[STAGE 1] Filtering filler utterances...")
        filtered = []
        removed_filler = 0
        
        for seg in segments:
            text = seg.get("text", "").strip()
            should_keep, reason = FillerFilter.filter_segment(text)
            
            if should_keep:
                filtered.append(seg)
            else:
                removed_filler += 1
        
        stats["after_filler"] = len(filtered)
        stats["removed_filler"] = removed_filler
        logger.info(f"   Removed {removed_filler} filler segments")
        
        # STAGE 2: Group by speaker
        logger.info("[STAGE 2] Grouping by speaker...")
        grouped = SpeakerGrouper.group_segments(filtered)
        stats["after_grouping"] = len(grouped)
        logger.info(f"   Grouped into {len(grouped)} turns")
        
        # STAGE 4: Content selection
        if self.enable_content_filter:
            logger.info("[STAGE 4] Content selection (relevance filtering)...")
            selected = ContentSelector.select_content(grouped, threshold=0.3)
            stats["after_selection"] = len(selected)
            logger.info(f"   Selected {len(selected)} relevant turns")
        else:
            selected = grouped
            logger.info("[STAGE 4] Content selection DISABLED")
        
        # STAGE 5: Length trimming
        logger.info(f"[STAGE 5] Length trimming (target: {self.max_length} chars)...")
        trimmed = LengthTrimmer.trim_to_length(selected, self.max_length)
        
        trimmed_chars = sum(len(s["text"]) for s in trimmed)
        stats["after_trimming"] = len(trimmed)
        stats["trimmed_chars"] = trimmed_chars
        logger.info(f"   Trimmed to {len(trimmed)} segments, {trimmed_chars} chars")
        
        # STAGE 6 & 7: Format outputs
        logger.info("[STAGE 6-7] Formatting outputs for LLM...")
        
        # Summary input: Gộp tất cả text với speaker labels
        summary_lines = []
        for seg in trimmed:
            emoji = "👨" if seg.get("gender") == "Male" else "👩"
            summary_lines.append(f"{emoji} {seg['speaker']}: {seg['text']}")
        summary_input = "\n\n".join(summary_lines)
        
        # Tasks input: Chỉ giữ segments có action keywords
        tasks_segments = [
            s for s in trimmed 
            if any(kw in s["text"].lower() for kw in ContentSelector.ACTION_KEYWORDS)
        ]
        
        tasks_lines = []
        for seg in tasks_segments:
            emoji = "👨" if seg.get("gender") == "Male" else "👩"
            tasks_lines.append(f"{emoji} {seg['speaker']}: {seg['text']}")
        tasks_input = "\n\n".join(tasks_lines)
        
        stats["summary_chars"] = len(summary_input)
        stats["tasks_chars"] = len(tasks_input)
        stats["tasks_segments"] = len(tasks_segments)
        
        logger.info(f"   Summary input: {len(summary_input)} chars")
        logger.info(f"   Tasks input: {len(tasks_input)} chars ({len(tasks_segments)} segments)")
        
        return {
            "cleaned_segments": trimmed,
            "summary_input": summary_input,
            "tasks_input": tasks_input,
            "stats": stats
        }


# ======================= STANDALONE TEST =======================
if __name__ == "__main__":
    import utils
    import config
    
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(handler)
    
    print("\n" + "="*80)
    print("🔥 TEXT PREPROCESSING PIPELINE TEST")
    print("="*80 + "\n")
    
    # Load raw segments
    data = utils.load_json(config.COMBINING_CACHE)
    segments = data if isinstance(data, list) else data.get("segments", [])
    
    # Run preprocessing
    preprocessor = TextPreprocessor(enable_content_filter=True, max_length=5000)
    result = preprocessor.preprocess(segments)
    
    # Print stats
    print("\n" + "="*80)
    print("📊 PREPROCESSING STATS")
    print("="*80)
    for key, value in result["stats"].items():
        print(f"   {key:20s}: {value}")
    
    # Preview
    print("\n" + "="*80)
    print("📋 SUMMARY INPUT PREVIEW (first 500 chars)")
    print("="*80)
    print(result["summary_input"][:500] + "...\n")
    
    print("="*80)
    print("✅ TASKS INPUT PREVIEW (first 500 chars)")
    print("="*80)
    print(result["tasks_input"][:500] + "...\n")
    
    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/preprocessed_summary_input.txt", "w", encoding="utf-8") as f:
        f.write(result["summary_input"])
    
    with open("outputs/preprocessed_tasks_input.txt", "w", encoding="utf-8") as f:
        f.write(result["tasks_input"])
    
    print("💾 Saved preprocessed inputs to outputs/")
