"""
Stage 7: Gender Classification
✅ GIẢM VAI TRÒ: Chỉ classify gender cho speakers metadata, không gán vào segments
"""
from typing import Dict, List
import numpy as np
import librosa
import soundfile as sf


class GenderClassifier:
    """
    Gender Classifier - Phân loại giới tính speakers
    
    ✅ SIMPLIFIED:
        - Chỉ classify cho speakers metadata
        - Không gán trực tiếp vào segments (segments sẽ lookup từ speakers)
        - Rule-based với fallback
    """
    
    def __init__(self):
        pass
    
    def classify(
        self,
        audio_path: str,
        speakers: Dict,
        segments: List[Dict]
    ) -> Dict[str, str]:
        """
        Classify gender cho từng speaker
        
        Args:
            audio_path: Path to audio
            speakers: Speakers metadata from diarization
            segments: Segments with speaker assignment
        
        Returns:
            Dict[str, str]: {speaker_id: gender}
        """
        print(f"   🎵 Analyzing audio for gender classification...")
        
        # Load audio
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        
        genders = {}
        
        for speaker_id in speakers.keys():
            print(f"   🎤 Processing {speaker_id}...")
            
            # Extract audio cho speaker này
            speaker_audio = self._extract_speaker_audio(
                waveform, sr, segments, speaker_id, max_duration=30
            )
            
            if speaker_audio is None or len(speaker_audio) < sr * 0.5:
                print(f"      ⚠️  Not enough audio → Unknown")
                genders[speaker_id] = "Unknown"
                continue
            
            # Extract pitch features
            pitch_features = self._extract_pitch(speaker_audio, sr)
            
            if pitch_features is None:
                genders[speaker_id] = "Unknown"
                continue
            
            # Classify
            gender = self._classify_from_pitch(pitch_features)
            genders[speaker_id] = gender
            
            emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
            print(f"      {emoji} {gender} (pitch: {pitch_features['mean']:.1f} Hz)")
        
        return genders
    
    def _extract_speaker_audio(
        self,
        waveform: np.ndarray,
        sr: int,
        segments: List[Dict],
        speaker_id: str,
        max_duration: float = 30
    ) -> np.ndarray:
        """Extract audio chunks for a specific speaker"""
        speaker_segments = [s for s in segments if s.get('speaker') == speaker_id]
        
        if not speaker_segments:
            return None
        
        chunks = []
        total_duration = 0
        
        for seg in speaker_segments[:10]:  # Lấy tối đa 10 segments đầu
            if total_duration >= max_duration:
                break
            
            start_sample = int(seg['start_time'] * sr)
            end_sample = int(seg['end_time'] * sr)
            
            chunk = waveform[start_sample:end_sample]
            chunks.append(chunk)
            
            total_duration += (end_sample - start_sample) / sr
        
        if not chunks:
            return None
        
        return np.concatenate(chunks)
    
    def _extract_pitch(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract pitch features"""
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=50, fmax=400)
            
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) < 10:
                return None
            
            return {
                'mean': np.mean(pitch_values),
                'median': np.median(pitch_values),
                'std': np.std(pitch_values)
            }
        except Exception as e:
            print(f"      ⚠️  Pitch extraction failed: {e}")
            return None
    
    def _classify_from_pitch(self, pitch_features: Dict) -> str:
        """Classify gender from pitch"""
        mean_pitch = pitch_features['mean']
        
        # Simple rule-based
        if mean_pitch < 155:
            return "Male"
        elif mean_pitch > 200:
            return "Female"
        else:
            # Ambiguous range, use median
            median_pitch = pitch_features['median']
            return "Female" if median_pitch > 175 else "Male"
