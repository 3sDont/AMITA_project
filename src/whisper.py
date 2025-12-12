"""
Optimized transcription với faster-whisper.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import soundfile as sf
from pydub import AudioSegment
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
import config
import torch
import utils
import numpy as np
import gc  # ✅ Thêm garbage collection

# ✅ IMPORT SHARED UTILITIES
from audio_chunker import AudioChunker, AudioChunk
from text_utils import (
    is_similar_text,
    clean_text,
    filter_spam_segments,
    calculate_text_entropy,
    calculate_word_repetition_ratio
)


def get_device():
    """Xác định device (GPU/CPU) dựa trên config"""
    print(f"\n🔍 Kiểm tra GPU...")
    print(f"   ⚙️  Config: USE_GPU = {config.USE_GPU}")
    print(f"   🖥️  CUDA available: {torch.cuda.is_available()}")
    
    if config.USE_GPU and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda
        
        print(f"\n✅ SỬ DỤNG GPU")
        print(f"   🎮 GPU: {gpu_name}")
        print(f"   💾 VRAM: {vram:.2f} GB")
        print(f"   🔧 CUDA: {cuda_version}")
        return device
    else:
        device = "cpu"
        
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

def transcribe_chunk_optimized(model, chunk_path, chunk_index, total_chunks):
    """
    Transcribe 1 chunk với faster-whisper.
    
    ✅ UNCHANGED: Giữ nguyên implementation
    """
    print(f"\n   📝 Xử lý chunk {chunk_index + 1}/{total_chunks}...")
    
    # ✅ faster-whisper API khác với openai-whisper
    segments, info = model.transcribe(
        chunk_path,
        language="vi",
        beam_size=5,
        vad_filter=True,  # ✅ Voice Activity Detection tự động
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Tối thiểu 500ms im lặng để cắt
            threshold=0.5,                # Độ nhạy phát hiện giọng nói
        ),
        word_timestamps=True,  # Timestamps từng từ
        condition_on_previous_text=True,
    )
    
    # Convert generator to list
    segments_list = []
    for segment in segments:
        segments_list.append({
            "start_time": segment.start,  # ✅ faster-whisper dùng .start không phải .start_time
            "end_time": segment.end,      # ✅ faster-whisper dùng .end không phải .end_time
            "text": segment.text,
            "no_speech_prob": segment.no_speech_prob,
            "words": [
                {"word": w.word, "start_time": w.start, "end_time": w.end, "probability": w.probability}
                for w in (segment.words or [])
            ] if segment.words else []
        })
    
    print(f"      ✅ Chunk {chunk_index + 1}: {len(segments_list)} segments")
    return {"segments": segments_list, "language": info.language}


def transcribe_audio_optimized(audio_path, chunk_duration_minutes=10):
    """
    ✅ MAIN API: Transcription tối ưu với faster-whisper.
    
    Changes:
        - Sử dụng AudioChunker để chia chunks (thống nhất với diarization)
        - Sử dụng text_utils.filter_spam_segments thay vì logic riêng
        - Giảm duplicate code
    """
    start_time = time.time()
    print("🎧 Đang chạy Whisper transcription (OPTIMIZED)...")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {audio_path}")
    
    device = get_device()
    
    # Checkpoint logic
    checkpoint_file = config.WHISPER_CACHE.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        checkpoint_data = utils.load_json(checkpoint_file)
        print(f"\n💾 Phát hiện checkpoint: {checkpoint_file}")
        print(f"   ✅ Đã xử lý {checkpoint_data.get('processed_chunks', 0)} chunks trước đó")
        
        user_input = input("   ❓ Tiếp tục từ checkpoint? (y/n): ").strip().lower()
        if user_input == 'y':
            all_segments = checkpoint_data.get('segments', [])
            processed_chunks = checkpoint_data.get('processed_chunks', 0)
            print(f"   ✅ Resume từ chunk {processed_chunks + 1}\n")
        else:
            all_segments = []
            processed_chunks = 0
            print("   ⚠️  Bắt đầu lại từ đầu\n")
    else:
        all_segments = []
        processed_chunks = 0
    
    # Load model
    print("   📥 Đang load Whisper model 'medium' (faster-whisper)...")
    load_start = time.time()
    
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        model = WhisperModel(
            "medium",
            device=device,
            compute_type=compute_type,
            cpu_threads=4 if device == "cpu" else 1,
            num_workers=1,
        )
        load_time = time.time() - load_start
        print(f"   ✅ Load model: {load_time:.1f}s (device={device}, compute_type={compute_type})")
        
    except RuntimeError as e:
        if "CUDA" in str(e) and device == "cuda":
            print(f"\n⚠️  LỖI GPU: {e}")
            print(f"\n🔄 FALLBACK: Thử load model với CPU...")
            
            device = "cpu"
            compute_type = "int8"
            
            model = WhisperModel(
                "medium",
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=1,
            )
            load_time = time.time() - load_start
            print(f"   ✅ Load model: {load_time:.1f}s (device=CPU, compute_type=int8)")
        else:
            raise
    
    # ✅ SỬ DỤNG AudioChunker
    print(f"   🎵 Đang xử lý file: {os.path.basename(audio_path)}")
    
    chunker = AudioChunker(audio_path)
    
    # ✅ Lấy chunks tối ưu (dùng diarization segments nếu có)
    diarization_segments = None
    if os.path.exists(config.DIARIZATION_CACHE):
        print(f"   🎯 Phát hiện diarization cache: {os.path.basename(config.DIARIZATION_CACHE)}")
        diar_data = utils.load_json(config.DIARIZATION_CACHE)
        diarization_segments = diar_data if isinstance(diar_data, list) else diar_data.get('segments', [])
    
    chunk_split_start = time.time()
    
    # ✅ THAY THẾ logic chia chunk cũ
    if chunker.duration / 60 > chunk_duration_minutes:
        print(f"   📌 File dài ({chunker.duration/60:.1f} phút) → Chunked processing")
        chunks = chunker.get_optimal_chunks(
            diarization_segments=diarization_segments,
            chunk_duration_minutes=chunk_duration_minutes
        )
    else:
        print(f"   ℹ️  File ngắn ({chunker.duration/60:.1f} phút), không cần chia chunks")
        chunks = [AudioChunk(
            start_time=0,
            end_time=chunker.duration,
            sample_rate=chunker.sample_rate,
            chunk_index=0
        )]
    
    chunk_time = time.time() - chunk_split_start
    print(f"   ✅ Chia chunks: {chunk_time:.1f}s\n")
    
    # Transcribe từng chunk
    trans_start = time.time()
    
    for chunk in chunks:
        # Skip chunks đã xử lý
        if chunk.chunk_index < processed_chunks:
            print(f"\n   ⏭️  Bỏ qua chunk {chunk.chunk_index+1}/{len(chunks)} (đã xử lý)\n")
            continue
        
        print(f"\n   📝 Xử lý chunk {chunk.chunk_index+1}/{len(chunks)} ({chunk.duration:.1f}s)...")
        
        chunk_trans_start = time.time()
        
        try:
            # ✅ Load audio cho chunk
            chunk.load_audio(audio_path)
            
            # ✅ Lưu chunk tạm (faster-whisper cần file path)
            temp_chunk_path = f"temp_whisper_chunk_{chunk.chunk_index:03d}.wav"
            chunk.save_to_file(temp_chunk_path)
            
            # Transcribe
            chunk_result = transcribe_chunk_optimized(model, temp_chunk_path, chunk.chunk_index, len(chunks))
            chunk_trans_time = time.time() - chunk_trans_start
            
            # ETA calculation
            if chunk.chunk_index < len(chunks) - 1:
                if chunk.chunk_index > 0:
                    avg_time_per_chunk = (time.time() - trans_start) / (chunk.chunk_index - processed_chunks + 1)
                    remaining_chunks = len(chunks) - chunk.chunk_index - 1
                    eta = avg_time_per_chunk * remaining_chunks
                    print(f"      ⏱️  Chunk time: {chunk_trans_time:.1f}s | ETA: {eta:.1f}s ({eta/60:.1f} min)")
            
            # Điều chỉnh timestamps
            for seg in chunk_result.get("segments", []):
                seg["start_time"] += chunk.start_time
                seg["end_time"] += chunk.start_time
                all_segments.append(seg)
            
            # Checkpoint
            checkpoint_data = {
                'processed_chunks': chunk.chunk_index + 1,
                'segments': all_segments,
                'timestamp': datetime.now().isoformat()
            }
            utils.save_json(checkpoint_data, checkpoint_file)
            
            # Cleanup
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
            
            # ✅ Giải phóng memory
            chunk.clear_audio_data()
            
        except Exception as e:
            print(f"      ❌ Lỗi chunk {chunk.chunk_index+1}: {e}")
            print(f"      🔄 Thử lại sau 3 giây...")
            time.sleep(3)
            
            try:
                chunk_result = transcribe_chunk_optimized(model, temp_chunk_path, chunk.chunk_index, len(chunks))
                for seg in chunk_result.get("segments", []):
                    seg["start_time"] += chunk.start_time
                    seg["end_time"] += chunk.start_time
                    all_segments.append(seg)
                print(f"      ✅ Retry thành công!")
            except Exception as retry_error:
                print(f"      ❌ Retry thất bại: {retry_error}")
                print(f"      ⚠️  Bỏ qua chunk này, tiếp tục...")
        
        # Garbage collection
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    trans_time = time.time() - trans_start
    print(f"\n   ✅ Transcription hoàn thành: {trans_time:.1f}s\n")
    
    # Clear GPU cache
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"   ✅ Tổng cộng phát hiện {len(all_segments)} segments")
    
    # ✅ SỬ DỤNG SHARED FILTER
    print("   🔍 Đang lọc spam, noise và repeats (ADVANCED)...")
    filter_start = time.time()
    
    filtered_segments, filter_stats = filter_spam_segments(
        all_segments,
        window_size=10
    )
    
    filter_time = time.time() - filter_start
    print(f"   ✅ Lọc hoàn thành: {filter_time:.1f}s\n")
    
    # Hiển thị thống kê
    print(f"   📊 Thống kê lọc spam:")
    print(f"      • Tổng segments:     {filter_stats['total']}")
    print(f"      • Đã loại bỏ:        {filter_stats['removed']} ({filter_stats['removed']/filter_stats['total']*100:.1f}%)")
    print(f"      • Còn lại:           {filter_stats['kept']} ({filter_stats['kept']/filter_stats['total']*100:.1f}%)")
    
    if filter_stats['reasons']:
        print(f"\n   🗑️  Chi tiết các lý do loại bỏ:")
        for reason, count in sorted(filter_stats['reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"      • {reason:25s}: {count:4d}")
    
    output = {
        "text": " ".join(seg.get("text", "") for seg in filtered_segments),
        "segments": filtered_segments,
        "language": "vi"
    }
    
    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n⏱️  THỜI GIAN CHI TIẾT:")
    print(f"   - Load model:       {load_time:>6.1f}s")
    print(f"   - Chia chunks:      {chunk_time:>6.1f}s")
    print(f"   - Transcription:    {trans_time:>6.1f}s")
    print(f"   - Filter:           {filter_time:>6.1f}s")
    print(f"   {'─'*35}")
    print(f"   🕓 TOTAL:           {elapsed:>6.1f}s ({elapsed/60:.1f} phút)")
    print(f"   ⏰ Kết thúc lúc: {end_dt}")
    
    total_text_length = sum(len(seg.get("text", "")) for seg in filtered_segments)
    print(f"\n   📝 Tổng số ký tự: {total_text_length:,}")
    if elapsed > 0:
        speedup = duration_minutes / (elapsed / 60)
        print(f"   🎯 Tốc độ xử lý: {speedup:.2f}x realtime")
    
    # Xóa checkpoint
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"\n   🗑️  Đã xóa checkpoint file")
        except:
            pass
    
    # Lưu cache
    utils.save_json(output, config.WHISPER_CACHE)
    return output


if __name__ == "__main__":
    """Chạy riêng giai đoạn 1: Whisper Transcription"""
    import sys
    
    try:
        print("\n" + "="*80)
        print("🎤 GIAI ĐOẠN 1: WHISPER TRANSCRIPTION (OPTIMIZED)")
        print("="*80 + "\n")
        
        # Kiểm tra file audio có tồn tại không
        if not os.path.exists(config.AUDIO_FILE):
            print(f"❌ Lỗi: Không tìm thấy file audio: {config.AUDIO_FILE}")
            print(f"💡 Vui lòng kiểm tra lại đường dẫn trong config.py")
            sys.exit(1)
        

        
        # Tạo thư mục outputs nếu chưa có
        os.makedirs("outputs", exist_ok=True)
        
        print(f"📁 Input:  {config.AUDIO_FILE}")
        print(f"⚙️  Config: USE_GPU={config.USE_GPU}\n")
        
        # Chạy transcription
        result = transcribe_audio_optimized(
            audio_path=config.AUDIO_FILE,
            chunk_length_minutes=10
        )

        print(f"\n✓ Kết quả đã lưu tại: {config.WHISPER_CACHE}")
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH GIAI ĐOẠN 1")
        print("="*80)
        print(f"📊 Số segments: {len(result['segments'])}")
        print(f"📝 Tổng text: {len(result['text'])} ký tự")
        print(f"🗣️  Ngôn ngữ: {result['language']}")
        
        # Preview một số segments đầu
        print(f"\n📋 Preview 3 segments đầu tiên:")
        print("-" * 80)
        for i, seg in enumerate(result['segments'][:3]):
            print(f"\n[{i+1}] {seg['start_time']:.2f}s → {seg['end_time']:.2f}s")
            print(f"    Text: {seg['text'][:100]}{'...' if len(seg['text']) > 100 else ''}")
        
        if len(result['segments']) > 3:
            print(f"\n... và {len(result['segments']) - 3} segments khác")
        
        print("\n" + "="*80)
        print("💡 TIP: Chạy tiếp giai đoạn 2 (diarization) nếu chưa có:")
        print("   python src/diarization.py")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
