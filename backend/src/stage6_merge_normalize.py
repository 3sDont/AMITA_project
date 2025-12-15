"""
Stage 6: Merge & Normalize
TẬP TRUNG TẤT CẢ LOGIC xử lý segments tại đây:
- Merge adjacent segments
- Remove duplicates
- Clean text (fillers)
- Normalize timestamps
- Format cho LLM
"""
from typing import Dict, List, Tuple
import re
from collections import Counter


class MergeNormalizer:
    """
    Merge & Normalizer - Xử lý tập trung tất cả logic
    
    Pipeline:
        1. Normalize format (chuẩn hóa keys)
        2. Merge adjacent segments (cùng speaker, gần nhau)
        3. Remove duplicates (phát hiện segments trùng)
        4. Clean text fillers (um, uh, à, ừ...)
        5. Normalize timestamps (làm tròn, validate)
        6. Quality check (loại bỏ segments kém chất lượng)
        7. Format for LLM (thêm metadata cần thiết)
    """
    
    def __init__(self):
        # Vietnamese fillers (từ lắp bắp cần loại bỏ)
        self.fillers = [
            'ừ', 'ừm', 'à', 'ạ', 'ơ', 'ồ', 'ô',
            'hừm', 'hừ', 'uhm', 'um', 'uh', 'eh',
            'thì', 'là', 'mà', 'này', 'đó'  # Filler words (nếu đứng một mình)
        ]
    
    def process(self, segments: List[Dict]) -> Dict:
        """
        Main processing pipeline
        
        Args:
            segments: Raw segments from speaker assignment
        
        Returns:
            Dict: {
                segments: List[Dict] - Final segments,
                stats: Dict - Processing statistics
            }
        """
        print(f"   📊 Input: {len(segments)} segments")
        
        stats = {
            'input_count': len(segments),
            'after_merge': 0,
            'after_dedup': 0,
            'after_clean': 0,
            'final_count': 0,
            'removed_by_quality': 0,
            'fillers_removed': 0
        }
        
        # Step 1: Normalize format
        print(f"   🔧 Step 1: Normalizing format...")
        normalized = self._normalize_format(segments)
        
        # Step 2: Merge adjacent segments
        print(f"   🔗 Step 2: Merging adjacent segments...")
        merged = self._merge_adjacent(normalized)
        stats['after_merge'] = len(merged)
        print(f"      ✅ {len(segments)} → {len(merged)} segments")
        
        # Step 3: Remove duplicates
        print(f"   🗑️  Step 3: Removing duplicates...")
        deduped = self._remove_duplicates(merged)
        stats['after_dedup'] = len(deduped)
        print(f"      ✅ {len(merged)} → {len(deduped)} segments")
        
        # Step 4: Clean text fillers
        print(f"   🧹 Step 4: Cleaning text fillers...")
        cleaned, fillers_count = self._clean_text_fillers(deduped)
        stats['after_clean'] = len(cleaned)
        stats['fillers_removed'] = fillers_count
        print(f"      ✅ Removed {fillers_count} filler words")
        
        # Step 5: Normalize timestamps
        print(f"   ⏱️  Step 5: Normalizing timestamps...")
        time_normalized = self._normalize_timestamps(cleaned)
        
        # Step 6: Quality check
        print(f"   ✅ Step 6: Quality check...")
        quality_checked = self._quality_check(time_normalized)
        removed = len(time_normalized) - len(quality_checked)
        stats['removed_by_quality'] = removed
        if removed > 0:
            print(f"      ⚠️  Removed {removed} low-quality segments")
        
        # Step 7: Format for LLM
        print(f"   📝 Step 7: Formatting for LLM...")
        final = self._format_for_llm(quality_checked)
        stats['final_count'] = len(final)
        
        print(f"\n   ✅ Processing complete: {len(segments)} → {len(final)} segments")
        
        return {
            'segments': final,
            'stats': stats
        }
    
    def _normalize_format(self, segments: List[Dict]) -> List[Dict]:
        """
        Step 1: Chuẩn hóa format
        Đảm bảo tất cả segments có cùng keys
        """
        normalized = []
        
        for seg in segments:
            # Chuẩn hóa time keys
            start = seg.get('start_time', seg.get('start', 0.0))
            end = seg.get('end_time', seg.get('end', 0.0))
            
            normalized_seg = {
                'start_time': float(start),
                'end_time': float(end),
                'speaker': seg.get('speaker', 'UNKNOWN'),
                'text': seg.get('text', '').strip(),
                'words': seg.get('words', []),
                'no_speech_prob': seg.get('no_speech_prob', 0.0)
            }
            
            normalized.append(normalized_seg)
        
        return normalized
    
    def _merge_adjacent(self, segments: List[Dict]) -> List[Dict]:
        """
        Step 2: Merge adjacent segments
        Gộp segments liền kề của cùng speaker
        """
        if not segments:
            return []
        
        # Sort by start time
        sorted_segs = sorted(segments, key=lambda x: x['start_time'])
        
        merged = []
        current = sorted_segs[0].copy()
        
        for seg in sorted_segs[1:]:
            gap = seg['start_time'] - current['end_time']
            same_speaker = (seg['speaker'] == current['speaker'])
            
            # Merge if: same speaker + gap < 2 seconds
            if same_speaker and gap < 2.0:
                # Extend time
                current['end_time'] = seg['end_time']
                
                # Merge text
                if current['text'] and seg['text']:
                    current['text'] += " " + seg['text']
                elif seg['text']:
                    current['text'] = seg['text']
                
                # Merge words
                current['words'].extend(seg['words'])
            else:
                merged.append(current)
                current = seg.copy()
        
        merged.append(current)
        return merged
    
    def _remove_duplicates(self, segments: List[Dict]) -> List[Dict]:
        """
        Step 3: Remove duplicate segments
        Phát hiện và loại bỏ segments trùng lặp
        """
        if not segments:
            return []
        
        deduped = []
        seen_texts = set()
        
        for seg in segments:
            text = seg['text'].lower().strip()
            
            # Skip nếu text rỗng
            if not text:
                continue
            
            # Check duplicate (exact match)
            if text in seen_texts:
                continue
            
            # Check similarity với segments gần đây
            is_similar = False
            for prev_seg in deduped[-5:]:  # Check 5 segments gần nhất
                prev_text = prev_seg['text'].lower().strip()
                
                # Simple similarity: check if one contains the other
                if (text in prev_text or prev_text in text) and len(text) > 10:
                    is_similar = True
                    break
            
            if not is_similar:
                deduped.append(seg)
                seen_texts.add(text)
        
        return deduped
    
    def _clean_text_fillers(self, segments: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Step 4: Clean text fillers
        Loại bỏ các từ lắp bắp (um, uh, à, ừ...)
        """
        total_fillers_removed = 0
        cleaned_segments = []
        
        for seg in segments:
            text = seg['text']
            
            # Remove fillers
            words = text.split()
            cleaned_words = []
            
            for word in words:
                # Remove punctuation for checking
                clean_word = re.sub(r'[^\w\s]', '', word).lower()
                
                # Check if filler
                if clean_word in self.fillers and len(words) > 2:
                    total_fillers_removed += 1
                    continue
                
                cleaned_words.append(word)
            
            # Reconstruct text
            cleaned_text = ' '.join(cleaned_words).strip()
            
            # Skip if text becomes empty
            if not cleaned_text:
                continue
            
            # Additional cleaning
            cleaned_text = self._additional_text_cleaning(cleaned_text)
            
            seg['text'] = cleaned_text
            cleaned_segments.append(seg)
        
        return cleaned_segments, total_fillers_removed
    
    def _additional_text_cleaning(self, text: str) -> str:
        """
        Làm sạch text thêm:
        - Normalize whitespace
        - Remove repeated punctuation
        - Fix spacing around punctuation
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove repeated punctuation (e.g., "......" -> "...")
        text = re.sub(r'([.!?,])\1{2,}', r'\1', text)
        
        # Fix spacing: "word,word" -> "word, word"
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        
        # Remove space before punctuation: "word ." -> "word."
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        return text.strip()
    
    def _normalize_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """
        Step 5: Normalize timestamps
        Làm tròn và validate timestamps
        """
        for seg in segments:
            # Round to 2 decimal places
            seg['start_time'] = round(seg['start_time'], 2)
            seg['end_time'] = round(seg['end_time'], 2)
            
            # Ensure end > start
            if seg['end_time'] <= seg['start_time']:
                seg['end_time'] = seg['start_time'] + 0.1
            
            # Calculate duration
            seg['duration'] = round(seg['end_time'] - seg['start_time'], 2)
        
        return segments
    
    def _quality_check(self, segments: List[Dict]) -> List[Dict]:
        """
        Step 6: Quality check
        Loại bỏ segments kém chất lượng
        """
        quality_segments = []
        
        for seg in segments:
            # Check 1: Text không rỗng
            if not seg['text'] or len(seg['text'].strip()) < 2:
                continue
            
            # Check 2: Duration hợp lý (0.1s - 120s)
            duration = seg['duration']
            if duration < 0.1 or duration > 120:
                continue
            
            # Check 3: Text/duration ratio hợp lý
            words = seg['text'].split()
            if duration > 0:
                words_per_second = len(words) / duration
                if words_per_second > 15 or words_per_second < 0.3:
                    # Quá nhanh (>15 words/s) hoặc quá chậm (<0.3 words/s)
                    continue
            
            # Check 4: No-speech probability không quá cao
            if seg['no_speech_prob'] > 0.9:
                continue
            
            # Check 5: Tối thiểu 1 từ có nghĩa
            alpha_chars = sum(c.isalpha() for c in seg['text'])
            if alpha_chars < 2:
                continue
            
            quality_segments.append(seg)
        
        return quality_segments
    
    def _format_for_llm(self, segments: List[Dict]) -> List[Dict]:
        """
        Step 7: Format cho LLM
        Thêm metadata cần thiết cho LLM processing
        """
        formatted = []
        
        for i, seg in enumerate(segments):
            formatted_seg = seg.copy()
            
            # Add segment index
            formatted_seg['segment_id'] = i
            
            # Add formatted time string (for display)
            formatted_seg['time_str'] = f"{self._format_time(seg['start_time'])} - {self._format_time(seg['end_time'])}"
            
            # Calculate word count
            formatted_seg['word_count'] = len(seg['text'].split())
            
            # Add speaker display name (with index)
            speaker = seg['speaker']
            if speaker.startswith('SPEAKER_'):
                speaker_num = speaker.split('_')[1]
                formatted_seg['speaker_display'] = f"Speaker {speaker_num}"
            else:
                formatted_seg['speaker_display'] = speaker
            
            formatted.append(formatted_seg)
        
        return formatted
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
