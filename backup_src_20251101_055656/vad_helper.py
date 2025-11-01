"""Voice Activity Detection helper."""
import torch

def detect_speech_regions(audio_path, threshold=0.5):
    """
    Phát hiện đoạn có giọng nói bằng Silero VAD.
    
    Args:
        audio_path: Đường dẫn audio
        threshold: Ngưỡng phát hiện (0.3-0.6)
    
    Returns:
        list: Danh sách timestamps có speech
    """
    print("🎤 Đang phát hiện đoạn có giọng nói...")
    
    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    # Load audio
    wav = read_audio(audio_path, sampling_rate=16000)
    
    # Phát hiện speech
    speech_timestamps = get_speech_timestamps(
        wav, 
        model,
        threshold=threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    )
    
    print(f"   ✅ Phát hiện {len(speech_timestamps)} đoạn có giọng nói")
    return speech_timestamps
