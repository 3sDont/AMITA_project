"""
Stage 3: Whisper Transcription Processor
Nhận chunks từ unified chunker và transcribe
"""
from typing import Dict, List
from faster_whisper import WhisperModel
import torch
import soundfile as sf
import numpy as np


class WhisperProcessor:
    """
    Whisper Processor - Transcribe audio với word-level timestamps
    
    Features:
        - Sử dụng faster-whisper (tốc độ cao)
        - Word-level timestamps (quan trọng cho speaker assignment)
        - VAD tích hợp
        - Batch processing với chunks
    """
    
    def __init__(self, model_size: str = "medium", device: str = "auto"):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size (tiny/base/small/medium/large)
            device: cuda/cpu/auto
        """
        self.model_size = model_size
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = None
    
    def _load_model(self):
        """Lazy load model khi cần"""
        if self.model is None:
            print(f"   📥 Loading Whisper model '{self.model_size}' on {self.device}...")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=4 if self.device == "cpu" else 1,
                    num_workers=1
                )
                print(f"      ✅ Model loaded")
            except RuntimeError as e:
                if "CUDA" in str(e) and self.device == "cuda":
                    print(f"      ⚠️  CUDA failed ({e}), falling back to CPU...")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        cpu_threads=4,
                        num_workers=1
                    )
                    print(f"      ✅ Model loaded on CPU")
                else:
                    raise
    
    def _transcribe_chunk(self, audio_data: np.ndarray, sr: int, chunk_info: Dict) -> Dict:
        """
        Transcribe một chunk
        
        Args:
            audio_data: Audio waveform (numpy array)
            sr: Sample rate
            chunk_info: Chunk metadata (start, end, chunk_id)
        
        Returns:
            Dict: {segments: [...], words: [...]}
        """
        # Save temp file (faster-whisper cần file path)
        import tempfile
        import os
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Write audio
            sf.write(temp_path, audio_data, sr)
            
            # Transcribe với word timestamps
            segments, info = self.model.transcribe(
                temp_path,
                language="vi",
                beam_size=5,
                vad_filter=True,  # ✅ VAD tự động
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5
                ),
                word_timestamps=True,  # ✅ QUAN TRỌNG cho speaker assignment
                condition_on_previous_text=True
            )
            
            # Collect results
            result_segments = []
            result_words = []
            
            chunk_start_offset = chunk_info['start']
            
            for seg in segments:
                # Adjust timestamps về timeline gốc
                seg_data = {
                    "start_time": seg.start + chunk_start_offset,
                    "end_time": seg.end + chunk_start_offset,
                    "text": seg.text.strip(),
                    "no_speech_prob": seg.no_speech_prob,
                    "chunk_id": chunk_info['chunk_id']
                }
                result_segments.append(seg_data)
                
                # Extract words với timestamps
                if seg.words:
                    for word in seg.words:
                        word_data = {
                            "word": word.word,
                            "start_time": word.start + chunk_start_offset,
                            "end_time": word.end + chunk_start_offset,
                            "probability": word.probability,
                            "chunk_id": chunk_info['chunk_id']
                        }
                        result_words.append(word_data)
            
            return {
                "segments": result_segments,
                "words": result_words,
                "language": info.language
            }
        
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def process(self, audio_path: str, chunks: List[Dict]) -> Dict:
        """
        Main processing function
        
        Args:
            audio_path: Path to preprocessed audio
            chunks: List of chunks from unified chunker
        
        Returns:
            Dict: {
                segments: List[Dict],
                words: List[Dict],
                language: str
            }
        """
        self._load_model()
        
        # Load audio một lần
        print(f"   📂 Loading audio: {audio_path}")
        waveform, sr = sf.read(audio_path)
        
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        
        all_segments = []
        all_words = []
        detected_language = None
        
        # Process từng chunk
        for i, chunk in enumerate(chunks):
            print(f"   🎤 Transcribing chunk {i+1}/{len(chunks)} [{chunk['start']:.1f}s - {chunk['end']:.1f}s]...")
            
            # Extract audio cho chunk này
            start_sample = int(chunk['start'] * sr)
            end_sample = int(chunk['end'] * sr)
            chunk_audio = waveform[start_sample:end_sample]
            
            # Transcribe
            result = self._transcribe_chunk(chunk_audio, sr, chunk)
            
            all_segments.extend(result['segments'])
            all_words.extend(result['words'])
            
            if detected_language is None:
                detected_language = result['language']
            
            print(f"      ✅ {len(result['segments'])} segments, {len(result['words'])} words")
        
        return {
            "segments": all_segments,
            "words": all_words,
            "language": detected_language
        }
