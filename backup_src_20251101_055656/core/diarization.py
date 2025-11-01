"""Speaker diarization using pyannote.audio."""
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch, os, time
import numpy as np
from dotenv import load_dotenv
import soundfile as sf

# Load biến môi trường từ file .env
load_dotenv()

def split_audio(input_path, chunk_length_ms=60000):
    """Chia file âm thanh lớn thành các đoạn nhỏ (1 phút)."""
    audio = AudioSegment.from_file(input_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"{input_path}_chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def diarize_audio(audio_path, output_path, num_speakers=None, min_speakers=None, max_speakers=None):
    """
    Phân đoạn người nói với pyannote.
    
    Args:
        num_speakers: Số người nói chính xác (nếu biết trước)
        min_speakers: Số người nói tối thiểu
        max_speakers: Số người nói tối đa
    """
    print("🔊 Bắt đầu diarization...")
    from datetime import datetime
    start_time = time.time()
    start_dt = datetime.now().strftime("%H:%M:%S")
    print(f"   ⏰ Bắt đầu lúc: {start_dt}")
    device = torch.device("cpu")
    print(f"⚙️  Thiết bị sử dụng: {device}")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("❌ Thiếu HF_TOKEN")
    
    # Load pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
    pipeline.to(device)

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
    
    # Trích xuất Annotation object từ DiarizeOutput
    if hasattr(result, 'speaker_diarization'):
        # Phiên bản mới: DiarizeOutput có attribute speaker_diarization
        diarization = result.speaker_diarization
        print(f"✅ Tìm thấy {len(list(diarization.itertracks()))} segments")
    elif hasattr(result, '__iter__'):
        # Phiên bản cũ: result là Annotation trực tiếp
        diarization = result
    else:
        raise ValueError(f"❌ Không nhận diện được format output: {type(result)}")
    
    # Đếm số người nói
    unique_speakers = set()
    with open(output_path, "w", encoding="utf-8") as f:
        segment_count = 0
        for segment, track, label in diarization.itertracks(yield_label=True):
            # ✅ FIX: Tăng độ chính xác từ .2f → .3f (mili-giây)
            f.write(f"{segment.start:.3f}s – {segment.end:.3f}s : {label}\n")
            segment_count += 1
            unique_speakers.add(label)
        
        if segment_count == 0:
            print("⚠️  CẢNH BÁO: Không phát hiện được người nói nào!")
            f.write("# No speakers detected\n")

    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    print(f"✅ Đã lưu {segment_count} segments vào {output_path}")
    print(f"👥 Số người nói phát hiện: {len(unique_speakers)} ({', '.join(sorted(unique_speakers))})")
    print(f"⏱️  Thời gian diarization: {elapsed:.2f}s ({elapsed/60:.1f} phút)")
    print(f"   ⏰ Kết thúc lúc: {end_dt}")
    return output_path
