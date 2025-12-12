"""
Stage 4: Speaker Diarization Processor
Nhận chunks từ unified chunker và diarize
"""
from typing import Dict, List
from pyannote.audio import Pipeline
import torch
import soundfile as sf
import numpy as np
import os


class DiarizationProcessor:
    """
    Diarization Processor - Phát hiện ai nói khi nào
    
    Output format:
        timeline: [{start, end, speaker}] - Simple timeline
        speakers: {speaker_id: metadata}
    """
    
    def __init__(self, hf_token: str = None):
        """
        Initialize diarization pipeline
        
        Args:
            hf_token: HuggingFace token (nếu None, lấy từ env)
        """
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("❌ HF_TOKEN required for diarization")
        
        self.hf_token = hf_token
        self.pipeline = None
        
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_pipeline(self):
        """Lazy load pipeline"""
        if self.pipeline is None:
            print(f"   📥 Loading pyannote pipeline on {self.device}...")
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                self.pipeline.to(self.device)
                
                # Tuning
                if hasattr(self.pipeline, 'clustering'):
                    self.pipeline.clustering.threshold = 0.71
                
                print(f"      ✅ Pipeline loaded")
            except RuntimeError as e:
                if "CUDA" in str(e) and self.device == torch.device("cuda"):
                    print(f"      ⚠️  CUDA failed ({e}), falling back to CPU...")
                    self.device = torch.device("cpu")
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self.hf_token
                    )
                    self.pipeline.to(self.device)
                    
                    # Tuning
                    if hasattr(self.pipeline, 'clustering'):
                        self.pipeline.clustering.threshold = 0.71
                    
                    print(f"      ✅ Pipeline loaded on CPU")
                else:
                    raise
    
    def _diarize_chunk(self, audio_data: np.ndarray, sr: int, chunk_info: Dict, **kwargs) -> List[Dict]:
        """
        Diarize một chunk
        
        Returns:
            List[Dict]: Timeline entries [{start, end, speaker}]
        """
        # Convert to tensor
        if audio_data.ndim == 1:
            audio_tensor = torch.from_numpy(audio_data[np.newaxis, :]).float()
        else:
            audio_tensor = torch.from_numpy(audio_data.T).float()
        
        audio_dict = {
            "waveform": audio_tensor,
            "sample_rate": sr
        }
        
        # Run diarization
        diarization = self.pipeline(audio_dict, **kwargs)
        
        # Convert to timeline
        timeline = []
        chunk_start_offset = chunk_info['start']
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                "start_time": float(turn.start) + chunk_start_offset,
                "end_time": float(turn.end) + chunk_start_offset,
                "speaker": speaker,
                "chunk_id": chunk_info['chunk_id']
            })
        
        return timeline
    
    def process(
        self,
        audio_path: str,
        chunks: List[Dict],
        min_speakers: int = None,
        max_speakers: int = None
    ) -> Dict:
        """
        Main processing function
        
        Args:
            audio_path: Path to preprocessed audio
            chunks: List of chunks from unified chunker
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        
        Returns:
            Dict: {
                timeline: List[{start, end, speaker}],
                speakers: {speaker_id: {total_time, segments_count}}
            }
        """
        self._load_pipeline()
        
        # Prepare kwargs
        kwargs = {}
        if min_speakers:
            kwargs['min_speakers'] = min_speakers
        if max_speakers:
            kwargs['max_speakers'] = max_speakers
        
        # Load audio
        print(f"   📂 Loading audio: {audio_path}")
        waveform, sr = sf.read(audio_path)
        
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        
        all_timeline = []
        
        # Process từng chunk
        for i, chunk in enumerate(chunks):
            print(f"   👥 Diarizing chunk {i+1}/{len(chunks)} [{chunk['start']:.1f}s - {chunk['end']:.1f}s]...")
            
            # Extract audio cho chunk này
            start_sample = int(chunk['start'] * sr)
            end_sample = int(chunk['end'] * sr)
            chunk_audio = waveform[start_sample:end_sample]
            
            # Diarize
            timeline = self._diarize_chunk(chunk_audio, sr, chunk, **kwargs)
            all_timeline.extend(timeline)
            
            print(f"      ✅ {len(timeline)} timeline entries")
        
        # Merge overlapping entries (cùng speaker, liền kề)
        print(f"   🔗 Merging overlapping entries...")
        merged_timeline = self._merge_timeline(all_timeline)
        print(f"      ✅ {len(merged_timeline)} entries after merge")
        
        # Build speakers metadata
        speakers = {}
        for entry in merged_timeline:
            speaker = entry['speaker']
            duration = entry['end_time'] - entry['start_time']
            
            if speaker not in speakers:
                speakers[speaker] = {
                    "total_time": 0,
                    "segments_count": 0
                }
            
            speakers[speaker]["total_time"] += duration
            speakers[speaker]["segments_count"] += 1
        
        return {
            "timeline": merged_timeline,
            "speakers": speakers
        }
    
    def _merge_timeline(self, timeline: List[Dict], gap_threshold: float = 0.5) -> List[Dict]:
        """Merge timeline entries cùng speaker, gần nhau"""
        if not timeline:
            return []
        
        # Sort by start time
        sorted_timeline = sorted(timeline, key=lambda x: x['start_time'])
        
        merged = []
        current = sorted_timeline[0].copy()
        
        for entry in sorted_timeline[1:]:
            # Cùng speaker và gap nhỏ
            if (entry['speaker'] == current['speaker'] and
                entry['start_time'] - current['end_time'] <= gap_threshold):
                # Merge
                current['end_time'] = entry['end_time']
            else:
                merged.append(current)
                current = entry.copy()
        
        merged.append(current)
        return merged
