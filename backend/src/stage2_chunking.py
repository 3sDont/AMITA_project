"""
Stage 2: Unified Chunking
Tạo chunk map dùng chung cho cả Whisper & Diarization

Strategy:
1. Ưu tiên VAD regions (không cắt giữa vùng có tiếng)
2. Fallback về silence detection
3. Cuối cùng mới time-based
"""
import numpy as np
from typing import List, Dict, Tuple, Optional


class UnifiedChunker:
    """
    Unified Chunker - Tạo chunks thống nhất cho toàn bộ pipeline
    
    Output format:
        [{
            "chunk_id": int,
            "start": float,  # seconds
            "end": float,
            "duration": float,
            "has_speech": bool
        }]
    """
    
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
    
    def create_chunks(
        self,
        chunk_duration_minutes: int = 10,
        vad_regions: Optional[List[Tuple[float, float]]] = None
    ) -> List[Dict]:
        """
        Create unified chunks
        
        Args:
            chunk_duration_minutes: Target duration per chunk
            vad_regions: VAD speech regions from preprocessing
        
        Returns:
            List of chunk dicts
        """
        # Get audio duration
        import soundfile as sf
        info = sf.info(self.audio_path)
        total_duration = info.duration
        
        target_duration = chunk_duration_minutes * 60
        
        # Strategy 1: VAD-based chunking (best)
        if vad_regions:
            print(f"   🎯 Using VAD-based chunking")
            return self._chunk_by_vad(total_duration, target_duration, vad_regions)
        
        # Strategy 2: Simple time-based (fallback)
        print(f"   ⏱️  Using time-based chunking")
        return self._chunk_by_time(total_duration, target_duration)
    
    def _chunk_by_vad(
        self,
        total_duration: float,
        target_duration: float,
        vad_regions: List[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Chia chunks dựa trên VAD regions - không cắt giữa speech
        """
        chunks = []
        current_start = 0.0
        chunk_id = 0
        
        while current_start < total_duration:
            target_end = min(current_start + target_duration, total_duration)
            
            # Tìm VAD region kết thúc gần target_end nhất
            best_end = target_end
            min_distance = float('inf')
            
            for region_start, region_end in vad_regions:
                # Chỉ xét regions trong khoảng hợp lý
                if current_start < region_end <= target_end + 60:
                    distance = abs(region_end - target_end)
                    if distance < min_distance:
                        min_distance = distance
                        best_end = region_end
            
            # Tạo chunk
            if best_end - current_start >= 30:  # Tối thiểu 30s
                chunks.append({
                    "chunk_id": chunk_id,
                    "start": current_start,
                    "end": best_end,
                    "duration": best_end - current_start,
                    "has_speech": True
                })
                chunk_id += 1
            
            current_start = best_end
        
        return chunks
    
    def _chunk_by_time(
        self,
        total_duration: float,
        target_duration: float
    ) -> List[Dict]:
        """
        Chia chunks theo thời gian cố định (fallback)
        """
        chunks = []
        current_start = 0.0
        chunk_id = 0
        
        while current_start < total_duration:
            chunk_end = min(current_start + target_duration, total_duration)
            
            chunks.append({
                "chunk_id": chunk_id,
                "start": current_start,
                "end": chunk_end,
                "duration": chunk_end - current_start,
                "has_speech": True  # Assume có speech
            })
            
            current_start = chunk_end
            chunk_id += 1
        
        return chunks
