"""Phát hiện phần có giọng nói để cải thiện diarization."""
import torch
import soundfile as sf

def detect_speech_regions(audio_path):
    """
    Sử dụng Silero VAD để phát hiện đoạn có giọng nói.
    Giúp loại bỏ im lặng và cải thiện độ chính xác.
    """
    print("🎤 Đang phát hiện đoạn có giọng nói...")
    
    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    # Load audio
    wav = read_audio(audio_path, sampling_rate=16000)
    
    # Phát hiện timestamps có giọng nói
    speech_timestamps = get_speech_timestamps(
        wav, 
        model,
        threshold=0.5,  # Độ nhạy (0.3-0.6 recommended)
        min_speech_duration_ms=250,  # Đoạn ngắn nhất coi là speech
        min_silence_duration_ms=100  # Đoạn yên lặng tối thiểu giữa 2 speech
    )
    
    print(f"✅ Phát hiện {len(speech_timestamps)} đoạn có giọng nói")
    return speech_timestamps
