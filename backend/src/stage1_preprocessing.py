"""
Stage 1: Audio Preprocessing
- Resample to 16kHz
- Normalize volume
- VAD (Voice Activity Detection) để detect vùng có tiếng
- High-pass filter
"""
import os
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from typing import Dict, List, Tuple, Optional

# ✅ Import pydub để hỗ trợ nhiều audio formats (MP3, M4A, AAC, etc.)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ✅ Load config values
try:
    from config import (
        SAMPLE_RATE,
        HIGH_PASS_FILTER_FREQ,
        NORMALIZE_TARGET,
        LARGE_FILE_THRESHOLD_MB
    )
except ImportError:
    # Fallback values
    SAMPLE_RATE = 16000
    HIGH_PASS_FILTER_FREQ = 80
    NORMALIZE_TARGET = 0.95
    LARGE_FILE_THRESHOLD_MB = 500


class AudioPreprocessor:
    """
    Audio Preprocessor với VAD
    
    Pipeline:
        1. Load audio
        2. Convert mono
        3. Resample 16kHz
        4. Normalize
        5. High-pass filter
        6. VAD (optional)
    """
    
    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        self.sample_rate = SAMPLE_RATE
    
    def process(self, enable_vad: bool = True) -> Dict:
        """
        Main processing pipeline
        
        Returns:
            Dict: {
                "output_path": str,
                "duration": float,
                "sample_rate": int,
                "vad_regions": List[Tuple[float, float]] (if enable_vad)
            }
        """
        print("   📂 Loading audio...")
        
        # ✅ Validate file exists and has content
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        
        file_size = os.path.getsize(self.audio_path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {self.audio_path}")
        
        if file_size < 1024:  # Less than 1KB
            raise ValueError(f"Audio file too small ({file_size} bytes), possibly corrupted: {self.audio_path}")
        
        print(f"   📊 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # ✅ For very large files, use streaming with librosa
        already_processed = False
        large_file_threshold = LARGE_FILE_THRESHOLD_MB * 1024 * 1024
        if file_size > large_file_threshold:
            print(f"   ⚠️  Large file detected ({file_size / 1024 / 1024:.0f}MB), using streaming mode...")
            waveform, sr = librosa.load(self.audio_path, sr=SAMPLE_RATE, mono=True)  # Direct to target SR
            print(f"   ✅ Loaded with librosa (already resampled to {SAMPLE_RATE}Hz, mono)")
            already_processed = True
        else:
            # ✅ Try soundfile first, fallback to pydub for unsupported formats
            try:
                waveform, sr = sf.read(self.audio_path)
            except Exception as e:
                if PYDUB_AVAILABLE:
                    print(f"   ⚠️  soundfile failed, using pydub for format conversion...")
                    waveform, sr = self._load_with_pydub(self.audio_path)
                else:
                    raise Exception(f"Cannot load audio: {e}. Install pydub for more format support.")
        
        # Convert stereo → mono (skip if already processed by librosa)
        if not already_processed and waveform.ndim > 1:
            print("   🔄 Converting stereo → mono...")
            waveform = np.mean(waveform, axis=1)
        
        # Resample to target sample rate using librosa (skip if already processed)
        if not already_processed and sr != SAMPLE_RATE:
            print(f"   🔄 Resampling {sr}Hz → {SAMPLE_RATE}Hz...")
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        # Normalize
        print("   🔊 Normalizing volume...")
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * NORMALIZE_TARGET
        
        # High-pass filter (remove low frequency noise)
        print(f"   🎛️  Applying high-pass filter ({HIGH_PASS_FILTER_FREQ}Hz)...")
        sos = signal.butter(4, HIGH_PASS_FILTER_FREQ, 'hp', fs=sr, output='sos')
        waveform = signal.sosfilt(sos, waveform)
        
        # Final normalize
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * NORMALIZE_TARGET
        
        # VAD (Voice Activity Detection)
        vad_regions = []
        if enable_vad:
            print("   🎤 Running VAD (Voice Activity Detection)...")
            vad_regions = self._detect_speech_regions(waveform, sr)
            print(f"      ✅ Found {len(vad_regions)} speech regions")
        
        # Save output
        output_path = self.audio_path.replace(".wav", "_preprocessed.wav")
        if not output_path.endswith("_preprocessed.wav"):
            output_path = output_path.rsplit(".", 1)[0] + "_preprocessed.wav"
        
        print(f"   💾 Saving preprocessed audio...")
        sf.write(output_path, waveform, sr)
        
        return {
            "output_path": output_path,
            "duration": len(waveform) / sr,
            "sample_rate": sr,
            "vad_regions": vad_regions
        }
    
    def _detect_speech_regions(
        self, 
        waveform: np.ndarray, 
        sr: int,
        frame_length_ms: int = 30,
        threshold_energy: float = 0.01
    ) -> List[Tuple[float, float]]:
        """
        Detect speech regions using energy-based VAD
        
        Returns:
            List of (start_time, end_time) tuples
        """
        frame_length = int(sr * frame_length_ms / 1000)
        hop_length = frame_length // 2
        
        # Calculate frame energy
        num_frames = (len(waveform) - frame_length) // hop_length + 1
        energies = []
        
        for i in range(num_frames):
            start_sample = i * hop_length
            end_sample = start_sample + frame_length
            frame = waveform[start_sample:end_sample]
            energy = np.sum(frame ** 2) / len(frame)
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Normalize energy
        max_energy = np.max(energies)
        if max_energy > 0:
            energies = energies / max_energy
        
        # Detect speech frames
        speech_frames = energies > threshold_energy
        
        # Merge consecutive speech frames into regions
        regions = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # Start of speech region
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech region
                start_time = start_frame * hop_length / sr
                end_time = i * hop_length / sr
                regions.append((start_time, end_time))
                in_speech = False
        
        # Handle last region
        if in_speech:
            start_time = start_frame * hop_length / sr
            end_time = len(waveform) / sr
            regions.append((start_time, end_time))
        
        return regions
    
    def _load_with_pydub(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio using pydub (supports MP3, M4A, AAC, OGG, etc.)
        
        Returns:
            Tuple[np.ndarray, int]: (waveform, sample_rate)
        """
        try:
            # Load with pydub
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            raise ValueError(
                f"Failed to decode audio file. The file may be corrupted or in an invalid format. "
                f"Error: {str(e)}\n"
                f"File: {audio_path}\n"
                f"Size: {os.path.getsize(audio_path)} bytes\n"
                f"Tip: Try re-uploading the file or converting it to WAV/MP3 format."
            )
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Get sample rate
        sr = audio.frame_rate
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1]
        if audio.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0
        
        return samples, sr
