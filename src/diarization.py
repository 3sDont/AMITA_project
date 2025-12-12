"""Speaker diarization using pyannote.audio."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
from pyannote.audio import Pipeline
import torch
import time
import numpy as np
import soundfile as sf
import config
import utils

# ✅ IMPORT SHARED UTILITIES
from audio_chunker import AudioChunker
from text_utils import is_similar_text  # ✅ Dùng từ text_utils thay vì define lại

def get_device():
    """Xác định device (GPU/CPU) dựa trên config"""
    print(f"\n🔍 Kiểm tra GPU...")
    print(f"   ⚙️  Config: USE_GPU = {config.USE_GPU}")
    print(f"   🖥️  CUDA available: {torch.cuda.is_available()}")
    
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        
        print(f"\n✅ SỬ DỤNG GPU")
        print(f"   🎮 GPU: {gpu_name}")
        print(f"   💾 VRAM: {vram:.2f} GB")
        print(f"   🔧 CUDA: {cuda_version}")
        return device
    else:
        device = torch.device("cpu")
        
        if config.USE_GPU and not torch.cuda.is_available():
            print(f"\n⚠️  GPU KHÔNG KHẢ DỤNG - FALLBACK VỀ CPU")
            print(f"\n💡 Để sử dụng GPU:")
            print(f"   1. Kiểm tra có GPU NVIDIA:")
            print(f"      → Mở Task Manager > Performance > GPU")
            print(f"   2. Cài lại PyTorch với CUDA:")
            print(f"      pip uninstall torch torchaudio")
            print(f"      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print(f"   3. Test GPU:")
            print(f"      python test_gpu.py")
        else:
            print(f"\n🖥️  SỬ DỤNG CPU (USE_GPU=False trong config)")
        
        return device

def merge_adjacent_segments(segments, gap_threshold=0.5):
    """
    Gộp các segments liền kề của cùng speaker.
    
    ✅ NOTE: Hàm này sẽ được move sang data_processor.py sau
    """
    if not segments:
        return []
    
    # Sort theo thời gian
    sorted_segs = sorted(segments, key=lambda x: x['start_time'])
    
    merged = []
    current = sorted_segs[0].copy()
    
    for seg in sorted_segs[1:]:
        # Cùng speaker và gần nhau → merge
        if (seg['speaker'] == current['speaker'] and 
            seg['start_time'] - current['end_time'] <= gap_threshold):
            current['end_time'] = seg['end_time']
        else:
            merged.append(current)
            current = seg.copy()
    
    merged.append(current)
    return merged


def diarize_audio_single(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
    """
    Diarization cho 1 file đơn (không chia chunks).
    
    ✅ CHANGED: Loại bỏ logic chunking, chỉ xử lý 1 file
    """
    device = get_device()
    print(f"⚙️  Thiết bị sử dụng: {device}")
    print(f"✓ File audio: {audio_path}")
    print("🔊 Bắt đầu diarization...")

    if not config.HF_TOKEN:
        raise ValueError("❌ Thiếu HF_TOKEN")
    
    # Load pipeline
    print("📥 Đang load pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=config.HF_TOKEN)
    pipeline.to(device)
    
    print(f"⏳ Đang phân đoạn người nói trên {device.type.upper()}...")

    # Tinh chỉnh tham số
    if hasattr(pipeline, 'clustering'):
        pipeline.clustering.threshold = 0.71
    
    if hasattr(pipeline, 'segmentation'):
        pipeline.segmentation.min_duration_off = 0.0
    
    # Load audio
    print("📂 Đang load audio file...")
    waveform, sample_rate = sf.read(audio_path)
    
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
    
    # Constraint số người nói
    kwargs = {}
    if num_speakers is not None:
        kwargs['num_speakers'] = num_speakers
    else:
        if min_speakers is not None:
            kwargs['min_speakers'] = min_speakers
        if max_speakers is not None:
            kwargs['max_speakers'] = max_speakers
    
    result = pipeline(audio_dict, **kwargs)
    
    # Xử lý kết quả
    diarization = getattr(result, "speaker_diarization", None)
    
    if diarization is None:
        if hasattr(result, "itertracks"):
            diarization = result
        else:
            raise ValueError(f"❌ Không nhận diện được format output: {type(result)}")
    
    # Chuyển đổi sang list
    segments = []
    speaker_times = {}
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.end - turn.start
        segments.append({
            "start_time": float(turn.start),
            "end_time": float(turn.end),
            "speaker": speaker
        })
        
        if speaker not in speaker_times:
            speaker_times[speaker] = 0
        speaker_times[speaker] += duration
    
    num_speakers = len(set(s['speaker'] for s in segments))
    total_duration = sum(s['end_time'] - s['start_time'] for s in segments)

    # Clear GPU cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\n✅ Hoàn thành!")
    print(f"   - Số phân đoạn: {len(segments)}")
    print(f"   - Số người nói: {num_speakers}")
    print(f"   - Tổng thời lượng: {total_duration:.1f}s ({total_duration/60:.1f} phút)")

    if speaker_times:
        print(f"\n   📊 Thống kê thời gian nói:")
        for speaker in sorted(speaker_times.keys()):
            time_spoken = speaker_times[speaker]
            percentage = (time_spoken / total_duration) * 100
            print(f"      {speaker}: {time_spoken:.1f}s ({percentage:.1f}%)")
    
    return segments


def diarize_audio_chunked(audio_path, chunk_duration_minutes=10, **kwargs):
    """
    Diarization theo chunks cho file dài.
    
    ✅ CHANGED: Sử dụng AudioChunker thay vì logic riêng
    """
    print(f"🔄 Chunked diarization mode (chunks: {chunk_duration_minutes} min)")
    
    # ✅ Sử dụng AudioChunker
    chunker = AudioChunker(audio_path)
    
    # Kiểm tra file có dài không
    if chunker.duration / 60 <= chunk_duration_minutes:
        print(f"   ℹ️  File ngắn ({chunker.duration/60:.1f} phút), không cần chia chunks")
        return diarize_audio_single(audio_path, **kwargs)
    
    # ✅ Lấy chunks (dùng silence detection tự động)
    chunks = chunker.chunk_by_silence(
        chunk_duration_minutes=chunk_duration_minutes,
        silence_threshold_db=30
    )
    
    print(f"   📊 Số chunks: {len(chunks)}\n")
    
    all_segments = []
    
    for chunk in chunks:
        print(f"\n   🎤 Chunk {chunk.chunk_index + 1}/{len(chunks)}: [{chunk.start_time:.1f}s - {chunk.end_time:.1f}s]")
        
        # ✅ Load audio cho chunk này
        chunk.load_audio(audio_path)
        
        # ✅ Lưu chunk tạm (pyannote cần file path)
        temp_chunk_path = f"temp_chunk_{chunk.chunk_index:03d}.wav"
        chunk.save_to_file(temp_chunk_path)
        
        try:
            # Diarize chunk
            chunk_segments = diarize_audio_single(temp_chunk_path, **kwargs)
            
            # Adjust timestamps
            for seg in chunk_segments:
                seg['start_time'] += chunk.start_time
                seg['end_time'] += chunk.start_time
            
            all_segments.extend(chunk_segments)
            print(f"      ✅ {len(chunk_segments)} segments")
            
        except Exception as e:
            print(f"      ❌ Lỗi: {e}")
        finally:
            # Cleanup
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
            
            # ✅ Giải phóng memory
            chunk.clear_audio_data()
    
    # Merge overlapping segments
    print(f"\n   🔗 Merging {len(all_segments)} segments...")
    merged = merge_adjacent_segments(all_segments, gap_threshold=0.5)
    print(f"      ✅ Còn lại {len(merged)} segments sau merge")
    
    return merged


def diarize_audio(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
    """
    ✅ MAIN API: Phân đoạn người nói với pyannote.
    
    Changes:
        - Sử dụng AudioChunker cho chunking logic
        - Auto-save kết quả
        - Tự động chọn strategy tối ưu
    """
    # Load audio info
    info = sf.info(audio_path)
    duration_minutes = info.duration / 60
    
    # ✅ Auto chunking cho file dài (>10 phút)
    if duration_minutes > 10:
        print(f"⚠️  File dài ({duration_minutes:.1f} phút) → Sử dụng chunked diarization")
        result = diarize_audio_chunked(
            audio_path,
            chunk_duration_minutes=8,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    else:
        result = diarize_audio_single(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    
    # ✅ AUTO-SAVE
    try:
        utils.save_json(result, config.DIARIZATION_CACHE)
        print(f"\n💾 Đã tự động lưu kết quả vào: {config.DIARIZATION_CACHE}")
    except Exception as e:
        print(f"\n⚠️  Không thể tự động lưu cache: {e}")
    
    return result


if __name__ == "__main__":
    """Chạy riêng giai đoạn 2"""
    try:
        # ✅ Lưu cache
        result = diarize_audio(
            config.AUDIO_FILE,
            min_speakers=config.MIN_SPEAKERS,
            max_speakers=config.MAX_SPEAKERS,
        )
        
        # Lưu vào cache file
        utils.save_json(result, config.DIARIZATION_CACHE)
        
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