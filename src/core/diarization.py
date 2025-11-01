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
    
    # ✅ Ước tính thời gian dựa trên độ dài audio
    import soundfile as sf
    audio_info = sf.info(audio_path)
    duration_minutes = audio_info.duration / 60
    estimated_time = duration_minutes * 0.5  # Trung bình 30s/phút audio
    print(f"   📊 Độ dài audio: {duration_minutes:.1f} phút")
    print(f"   ⏳ Ước tính: ~{estimated_time:.1f} phút\n")
    
    device = torch.device("cpu")
    print(f"⚙️  Thiết bị sử dụng: {device}")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("❌ Thiếu HF_TOKEN")
    
    # ✅ Load pipeline với progress
    print("📥 Đang load pipeline...")
    load_start = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN)
    pipeline.to(device)
    load_time = time.time() - load_start
    print(f"   ✅ Load pipeline: {load_time:.1f}s\n")

    # ✅ TINH CHỈNH THAM SỐ QUAN TRỌNG
    # Clustering: Giảm threshold để tách người nói tốt hơn
    if hasattr(pipeline, 'clustering'):
        pipeline.clustering.threshold = 0.71  # Default: 0.715 (càng thấp càng tách nhiều người)
    
    # Segmentation: Tăng độ nhạy phát hiện thay đổi người nói
    if hasattr(pipeline, 'segmentation'):
        pipeline.segmentation.min_duration_off = 0.0  # Cho phép đoạn im lặng ngắn
    
    print("📂 Đang load audio file...")
    audio_load_start = time.time()
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
    audio_load_time = time.time() - audio_load_start
    print(f"   ✅ Load audio: {audio_load_time:.1f}s\n")
    
    print("🎯 Đang chạy diarization...")
    diar_start = time.time()
    
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
    diar_time = time.time() - diar_start
    print(f"   ✅ Diarization hoàn thành: {diar_time:.1f}s\n")
    
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
    
    # ✅ Lưu kết quả với progress
    print("💾 Đang lưu kết quả...")
    save_start = time.time()
    
    unique_speakers = set()
    with open(output_path, "w", encoding="utf-8") as f:
        segment_count = 0
        for segment, track, label in diarization.itertracks(yield_label=True):
            f.write(f"{segment.start:.3f}s – {segment.end:.3f}s : {label}\n")
            segment_count += 1
            unique_speakers.add(label)
            
            # Progress bar mỗi 100 segments
            if segment_count % 100 == 0:
                print(f"   📝 Đã lưu {segment_count} segments...", end='\r')
        
        if segment_count == 0:
            print("⚠️  CẢNH BÁO: Không phát hiện được người nói nào!")
            f.write("# No speakers detected\n")
    
    save_time = time.time() - save_start
    print(f"\n   ✅ Lưu file: {save_time:.1f}s")

    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n✅ Đã lưu {segment_count} segments vào {output_path}")
    print(f"👥 Số người nói phát hiện: {len(unique_speakers)} ({', '.join(sorted(unique_speakers))})")
    print(f"\n⏱️  THỜI GIAN CHI TIẾT:")
    print(f"   - Load pipeline:  {load_time:>6.1f}s")
    print(f"   - Load audio:     {audio_load_time:>6.1f}s")
    print(f"   - Diarization:    {diar_time:>6.1f}s")
    print(f"   - Save results:   {save_time:>6.1f}s")
    print(f"   {'─'*35}")
    print(f"   🕓 TOTAL:         {elapsed:>6.1f}s ({elapsed/60:.1f} phút)")
    print(f"   ⏰ Kết thúc lúc: {end_dt}\n")
    return output_path
