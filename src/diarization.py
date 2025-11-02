"""Speaker diarization using pyannote.audio."""
import sys
import os
# Thêm thư mục gốc vào sys.path để import config và utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')  # Tắt warnings
from pyannote.audio import Pipeline
import torch
import time
import numpy as np
import soundfile as sf
import config
import utils

def get_device():
    """Xác định device (GPU/CPU) dựa trên config"""
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 Sử dụng GPU: {gpu_name}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   VRAM: {vram:.2f} GB")
    else:
        device = torch.device("cpu")
        if config.USE_GPU and not torch.cuda.is_available():
            print(f"⚠️  GPU không khả dụng, fallback về CPU")
        else:
            print(f"🖥️  Sử dụng CPU (USE_GPU=False)")
    return device

def diarize_audio(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
    """
    Phân đoạn người nói với pyannote.
    
    Args:
        audio_path: Đường dẫn file audio
        num_speakers: Số người nói chính xác (nếu biết trước)
        min_speakers: Số người nói tối thiểu
        max_speakers: Số người nói tối đa
    
    Returns:
        List[Dict]: Danh sách segments với speaker labels
    """
    device = get_device()
    print(f"⚙️  Thiết bị sử dụng: {device}")

    print(f"✓ File audio: {audio_path}")

    print("🔊 Bắt đầu diarization...")

    if not config.HF_TOKEN:
        raise ValueError("❌ Thiếu HF_TOKEN")
    
    # ✅ Load pipeline với progress
    print("📥 Đang load pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=config.HF_TOKEN)
    pipeline.to(device)
    # Chạy diarization với các tham số
    print(f"⏳ Đang phân đoạn người nói trên {device.type.upper()}...")


    # ✅ TINH CHỈNH THAM SỐ QUAN TRỌNG
    # Clustering: Giảm threshold để tách người nói tốt hơn
    if hasattr(pipeline, 'clustering'):
        pipeline.clustering.threshold = 0.71  # Default: 0.715 (càng thấp càng tách nhiều người)
    
    # Segmentation: Tăng độ nhạy phát hiện thay đổi người nói
    if hasattr(pipeline, 'segmentation'):
        pipeline.segmentation.min_duration_off = 0.0  # Cho phép đoạn im lặng ngắn
    
    print("📂 Đang load audio file...")
    waveform, sample_rate = sf.read(audio_path)
    
    # Convert sang tensor
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    else:
        waveform = waveform.T
    
    waveform_tensor = torch.from_numpy(waveform).float()
    
    audio_dict = {
        "waveform": waveform_tensor,
        "sample_rate": sample_rate
    }
    
    print("🎯 Đang chạy diarization...")

    
    # ✅ THÊM CONSTRAINT SỐ NGƯỜI NÓI
    kwargs = {}
    if num_speakers is not None:
        kwargs['num_speakers'] = num_speakers
        print(f"   🎤 Số người nói cố định: {num_speakers}")
    else:
        if min_speakers is not None:
            kwargs['min_speakers'] = min_speakers
            print(f"   🎤 Số người nói tối thiểu: {min_speakers}")
        if max_speakers is not None:
            kwargs['max_speakers'] = max_speakers
            print(f"   🎤 Số người nói tối đa: {max_speakers}")
    
    result = pipeline(audio_dict, **kwargs)
    
    # ✅ FIX: Xử lý DiarizeOutput object đúng cách
    print(f"📊 Kiểu kết quả: {type(result)}")
    
    # Trích xuất Annotation object từ DiarizeOutput (an toàn hơn)
    diarization = getattr(result, "speaker_diarization", None)
    
    if diarization is None:
        # Nếu không có speaker_diarization, kiểm tra xem result có itertracks không
        if hasattr(result, "itertracks"):
            diarization = result
        else:
            raise ValueError(f"❌ Không nhận diện được format output: {type(result)}")
    
    # Đếm segments an toàn (không convert toàn bộ sang list)
    num_segments = sum(1 for _ in diarization.itertracks(yield_label=True))
    print(f"✅ Tìm thấy {num_segments} segments")
    
    # Chuyển đổi sang list và tính thống kê
    segments = []
    speaker_times = {}  # Tracking thời gian nói của mỗi speaker
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        segments.append({
            "start_time": float(turn.start),
            "end_time": float(turn.end),
            "speaker": speaker
        })
        
        # Tính tổng thời gian nói của mỗi speaker
        if speaker not in speaker_times:
            speaker_times[speaker] = 0
        speaker_times[speaker] += duration
    
    num_speakers = len(set(s['speaker'] for s in segments))
    total_duration = sum(s['end_time'] - s['start_time'] for s in segments)

    # Clear GPU cache nếu dùng GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\n✅ Hoàn thành!")
    print(f"   - Device: {device.type.upper()}")
    print(f"   - Số phân đoạn: {len(segments)}")
    print(f"   - Số người nói: {num_speakers}")
    print(f"   - Tổng thời lượng phát biểu: {total_duration:.1f}s ({total_duration/60:.1f} phút)")

    # Hiển thị thống kê từng speaker
    if speaker_times:
        print(f"\n   📊 Thống kê thời gian nói:")
        for speaker in sorted(speaker_times.keys()):
            time_spoken = speaker_times[speaker]
            percentage = (time_spoken / total_duration) * 100
            print(f"      {speaker}: {time_spoken:.1f}s ({percentage:.1f}%)")
    
    # Lưu cache
    utils.save_json(segments, config.DIARIZATION_CACHE)
    
    return segments


if __name__ == "__main__":
    """Chạy riêng giai đoạn 2"""
    try:
        # Chỉ truyền các tham số được hỗ trợ
        result = diarize_audio(
            config.AUDIO_FILE,
            min_speakers=config.MIN_SPEAKERS,
            max_speakers=config.MAX_SPEAKERS,
        )
        print(f"\n✓ Kết quả đã lưu tại: {config.DIARIZATION_CACHE}")
        
        # In preview chi tiết
        print("\n" + "="*70)
        print("📄 PREVIEW 5 SEGMENT ĐẦU TIÊN")
        print("="*70)
        for i, seg in enumerate(result[:5]):
            duration = seg['end_time'] - seg['start_time']
            print(f"\n[{i+1}] {seg['speaker']}")
            print(f"    ⏱️  Thời gian: {seg['start_time']:.2f}s → {seg['end_time']:.2f}s (dài {duration:.1f}s)")
        
        if len(result) > 5:
            print(f"\n... và {len(result) - 5} segment khác")
        print("="*70)
        
        # Hiển thị lời khuyên nếu kết quả không như mong đợi
        num_speakers = len(set(s['speaker'] for s in result))
        print(f"\n💡 TIPS:")
        if config.MIN_SPEAKERS and num_speakers < config.MIN_SPEAKERS:
            print(f"   ⚠️  Phát hiện {num_speakers} người nói, ít hơn MIN_SPEAKERS={config.MIN_SPEAKERS}")
            print(f"   → Thử giảm MIN_SPEAKERS hoặc để None để auto-detect")
        elif config.MAX_SPEAKERS and num_speakers > config.MAX_SPEAKERS:
            print(f"   ⚠️  Phát hiện {num_speakers} người nói, nhiều hơn MAX_SPEAKERS={config.MAX_SPEAKERS}")
            print(f"   → Thử tăng MAX_SPEAKERS hoặc để None để auto-detect")
        else:
            print(f"   ✓ Số người nói phát hiện ({num_speakers}) nằm trong khoảng mong đợi")
            print(f"   → Nếu muốn điều chỉnh thêm, hãy:")
            print(f"      • Cải thiện chất lượng audio (giảm noise)")
            print(f"      • Đặt chính xác MIN_SPEAKERS = MAX_SPEAKERS nếu biết số người")
            print(f"      • Dùng model mới hơn như speaker-diarization-3.2 (nếu có)")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()