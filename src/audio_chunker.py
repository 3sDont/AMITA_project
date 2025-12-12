"""
Audio Chunker - Xử lý chunking audio tập trung
Mục đích: 
    - Loại bỏ duplicate chunking logic giữa diarization.py và whisper.py
    - Giảm IO (không ghi nhiều file chunk trung gian)
    - Cung cấp API thống nhất cho chunking
"""
import os
import warnings
warnings.filterwarnings('ignore')
from typing import List, Tuple, Optional, Dict
import soundfile as sf
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """
    Data class đại diện cho 1 audio chunk.
    
    Attributes:
        start_time: Thời gian bắt đầu (giây)
        end_time: Thời gian kết thúc (giây)
        audio_data: Numpy array chứa waveform (optional, để tiết kiệm RAM)
        sample_rate: Sample rate (Hz)
        chunk_index: Index của chunk trong sequence
    """
    start_time: float
    end_time: float
    audio_data: Optional[np.ndarray] = None
    sample_rate: int = 16000
    chunk_index: int = 0
    
    @property
    def duration(self) -> float:
        """Tính thời lượng chunk (giây)."""
        return self.end_time - self.start_time
    
    def load_audio(self, audio_path: str):
        """
        Load audio data cho chunk từ file gốc.
        
        Args:
            audio_path: Đường dẫn file audio gốc
        
        Note: Chỉ load đúng phần cần thiết (tiết kiệm RAM)
        """
        if self.audio_data is not None:
            return  # Đã load rồi
        
        # Tính vị trí trong file
        start_sample = int(self.start_time * self.sample_rate)
        num_samples = int(self.duration * self.sample_rate)
        
        # Load chỉ phần cần thiết
        with sf.SoundFile(audio_path) as f:
            f.seek(start_sample)
            self.audio_data = f.read(num_samples)
            
            # Convert stereo → mono nếu cần
            if self.audio_data.ndim > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
    
    def save_to_file(self, output_path: str):
        """
        Lưu chunk ra file (nếu cần).
        
        Args:
            output_path: Đường dẫn file output
        """
        if self.audio_data is None:
            raise ValueError("Audio data chưa được load")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        sf.write(output_path, self.audio_data, self.sample_rate)
    
    def clear_audio_data(self):
        """Giải phóng memory của audio data."""
        self.audio_data = None


class AudioChunker:
    """
    Audio Chunker - Chia audio thành chunks một cách thông minh.
    
    Strategies:
        1. time_based: Chia theo thời gian cố định
        2. silence_based: Chia theo điểm im lặng (dùng librosa)
        3. segment_based: Chia theo diarization segments
    """
    
    def __init__(self, audio_path: str):
        """
        Khởi tạo chunker.
        
        Args:
            audio_path: Đường dẫn file audio cần chia
        """
        self.audio_path = audio_path
        
        # Load audio info (không load toàn bộ audio vào RAM)
        self.info = sf.info(audio_path)
        self.duration = self.info.duration
        self.sample_rate = self.info.samplerate
        
        print(f"📂 AudioChunker initialized:")
        print(f"   - File: {os.path.basename(audio_path)}")
        print(f"   - Duration: {self.duration/60:.1f} min")
        print(f"   - Sample rate: {self.sample_rate} Hz")
    
    def chunk_by_time(
        self,
        chunk_duration_minutes: int = 10,
        overlap_seconds: float = 0.0
    ) -> List[AudioChunk]:
        """
        Chia audio theo thời gian cố định.
        
        Args:
            chunk_duration_minutes: Độ dài mỗi chunk (phút)
            overlap_seconds: Độ overlap giữa các chunks (giây)
        
        Returns:
            List[AudioChunk]: Danh sách chunks
        
        Use case: Khi không có thông tin segments/silence
        """
        chunk_duration_sec = chunk_duration_minutes * 60
        chunks = []
        
        current_start = 0.0
        chunk_idx = 0
        
        while current_start < self.duration:
            # Xác định end time
            chunk_end = min(current_start + chunk_duration_sec, self.duration)
            
            # Tạo chunk (chưa load audio)
            chunk = AudioChunk(
                start_time=current_start,
                end_time=chunk_end,
                sample_rate=int(self.sample_rate),
                chunk_index=chunk_idx
            )
            chunks.append(chunk)
            
            # Move đến chunk tiếp theo (có overlap)
            current_start = chunk_end - overlap_seconds
            chunk_idx += 1
        
        print(f"   ✂️  Time-based chunking: {len(chunks)} chunks")
        return chunks
    
    def chunk_by_silence(
        self,
        chunk_duration_minutes: int = 10,
        silence_threshold_db: int = 30,
        search_window_seconds: int = 60
    ) -> List[AudioChunk]:
        """
        Chia audio theo điểm im lặng (silence detection).
        
        Args:
            chunk_duration_minutes: Target duration cho mỗi chunk
            silence_threshold_db: Ngưỡng dB để coi là im lặng
            search_window_seconds: Khoảng thời gian tìm kiếm silence gần target
        
        Returns:
            List[AudioChunk]: Danh sách chunks
        
        Use case: Khi muốn chia tự nhiên theo pauses trong speech
        
        Note: Yêu cầu librosa
        """
        try:
            import librosa
        except ImportError:
            print("   ⚠️  librosa not found, fallback to time-based chunking")
            return self.chunk_by_time(chunk_duration_minutes)
        
        print("   🔍 Detecting silence boundaries...")
        
        # Load audio để phân tích silence
        y, sr = librosa.load(self.audio_path, sr=16000, mono=True)
        
        # Phát hiện non-silent intervals
        intervals = librosa.effects.split(y, top_db=silence_threshold_db)
        
        # Convert sang timestamps
        silence_times = []
        for i in range(len(intervals) - 1):
            end_current = intervals[i][1] / sr
            start_next = intervals[i+1][0] / sr
            silence_mid = (end_current + start_next) / 2
            silence_times.append(silence_mid)
        
        print(f"      ✅ Found {len(silence_times)} silence points")
        
        # Tạo chunks dựa trên silence
        chunks = []
        current_start = 0.0
        chunk_idx = 0
        target_duration = chunk_duration_minutes * 60
        
        while current_start < self.duration:
            target_end = current_start + target_duration
            
            # Tìm silence gần nhất với target_end
            best_silence = None
            min_distance = float('inf')
            
            for silence_time in silence_times:
                # Chỉ xét silence trong khoảng hợp lý
                if current_start < silence_time <= target_end + search_window_seconds:
                    distance = abs(silence_time - target_end)
                    if distance < min_distance:
                        min_distance = distance
                        best_silence = silence_time
            
            # Xác định điểm cắt
            chunk_end = best_silence if best_silence else min(target_end, self.duration)
            
            # Tạo chunk (tránh chunks quá ngắn)
            if chunk_end - current_start >= 30:
                chunk = AudioChunk(
                    start_time=current_start,
                    end_time=chunk_end,
                    sample_rate=int(self.sample_rate),
                    chunk_index=chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1
            
            current_start = chunk_end
        
        print(f"   ✂️  Silence-based chunking: {len(chunks)} chunks")
        return chunks
    
    def chunk_by_segments(
        self,
        segments: List[Dict],
        chunk_duration_minutes: int = 10
    ) -> List[AudioChunk]:
        """
        Chia audio theo diarization segments.
        
        Args:
            segments: List of diarization segments (có keys: start_time, end_time, speaker)
            chunk_duration_minutes: Target duration
        
        Returns:
            List[AudioChunk]: Danh sách chunks
        
        Use case: Sau khi có diarization, chia chunks không cắt đứng segment
        
        Ưu điểm: Không bao giờ cắt đứng giữa một lượt phát biểu
        """
        if not segments:
            print("   ⚠️  No segments provided, fallback to time-based chunking")
            return self.chunk_by_time(chunk_duration_minutes)
        
        print("   🎯 Segment-based chunking...")
        
        # Sort segments theo thời gian
        sorted_segments = sorted(segments, key=lambda x: x.get('start_time', x.get('start', 0)))
        
        chunks = []
        current_start = 0.0
        chunk_idx = 0
        target_duration = chunk_duration_minutes * 60
        
        while current_start < self.duration:
            target_end = current_start + target_duration
            
            # Tìm segment cuối cùng nằm trong target range
            best_segment = None
            for seg in sorted_segments:
                seg_start = seg.get('start_time', seg.get('start', 0))
                seg_end = seg.get('end_time', seg.get('end', 0))
                
                # Segment bắt đầu sau current_start và kết thúc trước/gần target_end
                if seg_start >= current_start and seg_end <= target_end + 60:
                    best_segment = seg
                elif seg_start > target_end + 60:
                    break  # Đã quá xa
            
            # Xác định điểm cắt
            if best_segment:
                chunk_end = best_segment.get('end_time', best_segment.get('end', 0))
            else:
                chunk_end = min(target_end, self.duration)
            
            # Tạo chunk (tránh quá ngắn)
            if chunk_end - current_start >= 30:
                chunk = AudioChunk(
                    start_time=current_start,
                    end_time=chunk_end,
                    sample_rate=int(self.sample_rate),
                    chunk_index=chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1
            
            current_start = chunk_end
        
        print(f"   ✂️  Segment-based chunking: {len(chunks)} chunks")
        return chunks
    
    def get_optimal_chunks(
        self,
        diarization_segments: Optional[List[Dict]] = None,
        chunk_duration_minutes: int = 10
    ) -> List[AudioChunk]:
        """
        Lấy chunks với strategy tối ưu nhất.
        
        Logic:
            1. Nếu có diarization segments → dùng segment-based
            2. Nếu không, thử silence-based (nếu có librosa)
            3. Fallback về time-based
        
        Args:
            diarization_segments: Diarization segments (optional)
            chunk_duration_minutes: Target duration
        
        Returns:
            List[AudioChunk]: Optimal chunks
        """
        # Strategy 1: Segment-based (best)
        if diarization_segments:
            print("   🎯 Using segment-based chunking (optimal)")
            return self.chunk_by_segments(diarization_segments, chunk_duration_minutes)
        
        # Strategy 2: Silence-based (good)
        try:
            import librosa
            print("   🔇 Using silence-based chunking")
            return self.chunk_by_silence(chunk_duration_minutes)
        except ImportError:
            pass
        
        # Strategy 3: Time-based (fallback)
        print("   ⏱️  Using time-based chunking (fallback)")
        return self.chunk_by_time(chunk_duration_minutes)
