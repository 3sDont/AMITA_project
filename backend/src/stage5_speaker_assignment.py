"""
Stage 5: Speaker Assignment
Gán speaker cho mỗi segment dựa trên word-level voting

✅ KHÔNG DÙNG text similarity nữa (không chính xác)
✅ SỬ DỤNG word-level timestamps + time overlap voting
"""
from typing import Dict, List
from collections import Counter


class SpeakerAssigner:
    """
    Speaker Assigner - Gán speaker cho segments dựa trên word-level voting
    
    Logic:
        1. Với mỗi word trong segment, check overlap với diarization timeline
        2. Vote speaker dựa trên số lượng words overlap
        3. Speaker có nhiều votes nhất → gán cho segment đó
    """
    
    def __init__(self):
        pass
    
    def _get_speaker_for_word(self, word: Dict, diarization_timeline: List[Dict]) -> str:
        """
        Tìm speaker cho 1 word dựa trên time overlap
        
        Args:
            word: {word, start_time, end_time, probability}
            diarization_timeline: [{start_time, end_time, speaker}]
        
        Returns:
            str: Speaker ID hoặc "UNKNOWN"
        """
        word_start = word['start_time']
        word_end = word['end_time']
        
        max_overlap = 0
        best_speaker = "UNKNOWN"
        
        for entry in diarization_timeline:
            diar_start = entry['start_time']
            diar_end = entry['end_time']
            
            # Calculate overlap
            overlap_start = max(word_start, diar_start)
            overlap_end = min(word_end, diar_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = entry['speaker']
        
        return best_speaker
    
    def _assign_speaker_to_segment(
        self,
        segment: Dict,
        segment_words: List[Dict],
        diarization_timeline: List[Dict]
    ) -> str:
        """
        Gán speaker cho segment dựa trên word-level voting
        
        Args:
            segment: Whisper segment
            segment_words: List words thuộc segment này
            diarization_timeline: Diarization timeline
        
        Returns:
            str: Speaker ID
        """
        if not segment_words:
            # Fallback: dùng segment-level overlap
            seg_start = segment['start_time']
            seg_end = segment['end_time']
            
            max_overlap = 0
            best_speaker = "UNKNOWN"
            
            for entry in diarization_timeline:
                diar_start = entry['start_time']
                diar_end = entry['end_time']
                
                overlap_start = max(seg_start, diar_start)
                overlap_end = min(seg_end, diar_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = entry['speaker']
            
            return best_speaker
        
        # Word-level voting
        speaker_votes = []
        
        for word in segment_words:
            speaker = self._get_speaker_for_word(word, diarization_timeline)
            if speaker != "UNKNOWN":
                speaker_votes.append(speaker)
        
        if not speaker_votes:
            return "UNKNOWN"
        
        # Speaker có nhiều votes nhất
        speaker_counts = Counter(speaker_votes)
        best_speaker = speaker_counts.most_common(1)[0][0]
        
        return best_speaker
    
    def assign(
        self,
        whisper_segments: List[Dict],
        whisper_words: List[Dict],
        diarization_timeline: List[Dict]
    ) -> List[Dict]:
        """
        Main assignment function
        
        Args:
            whisper_segments: Segments from Whisper
            whisper_words: Words from Whisper (with timestamps)
            diarization_timeline: Timeline from diarization
        
        Returns:
            List[Dict]: Segments with speaker assigned
        """
        print(f"   🎯 Assigning speakers using word-level voting...")
        
        assigned_segments = []
        
        for seg in whisper_segments:
            seg_start = seg['start_time']
            seg_end = seg['end_time']
            
            # Tìm words thuộc segment này
            segment_words = [
                w for w in whisper_words
                if seg_start <= w['start_time'] < seg_end
            ]
            
            # Assign speaker
            speaker = self._assign_speaker_to_segment(seg, segment_words, diarization_timeline)
            
            # Create output segment
            assigned_seg = {
                "start_time": seg_start,
                "end_time": seg_end,
                "text": seg['text'],
                "speaker": speaker,
                "words": segment_words,  # Keep words for further processing
                "no_speech_prob": seg.get('no_speech_prob', 0)
            }
            
            assigned_segments.append(assigned_seg)
        
        # Stats
        speaker_counts = Counter(seg['speaker'] for seg in assigned_segments)
        print(f"      ✅ Assignment complete:")
        for speaker, count in speaker_counts.most_common():
            print(f"         {speaker}: {count} segments")
        
        return assigned_segments
