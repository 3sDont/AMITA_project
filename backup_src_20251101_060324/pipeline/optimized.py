"""
Optimized pipeline với caching và parallel processing.
"""
import sys
import time
import os
from datetime import datetime
import json

# Import modules
from utils import prepare_output_paths
from diarize import diarize_audio
from gender_classify import classify_gender
from combine import combine_results
from audio_preprocess import enhance_audio
from cache_manager import load_cache, save_cache, clear_cache

# Import transcription modules
try:
    from transcribe_optimized import transcribe_audio_optimized
    HAS_OPTIMIZED = True
except ImportError:
    print("⚠️  faster-whisper not installed, using standard whisper")
    from transcribe import transcribe_audio as transcribe_audio_optimized
    HAS_OPTIMIZED = False

try:
    from transcribe_parallel import transcribe_audio_parallel
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False


def convert_to_audio(input_path):
    """Convert video sang audio nếu cần."""
    from pydub import AudioSegment
    
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
        print(f"🎬 Phát hiện file video ({ext}), đang convert sang audio...")
        output_path = input_path.rsplit('.', 1)[0] + '_audio.wav'
        
        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format='wav')
            print(f"✅ Đã convert sang: {output_path}\n")
            return output_path
        except Exception as e:
            print(f"❌ Lỗi convert video: {e}")
            raise
    
    elif ext in ['.mp3', '.m4a', '.aac', '.ogg', '.flac']:
        print(f"🎵 Phát hiện file audio ({ext}), giữ nguyên để xử lý sau...")
        return input_path
    
    elif ext == '.wav':
        print(f"✅ File WAV, không cần convert")
        return input_path
    
    else:
        print(f"⚠️  Format không nhận diện ({ext}), thử xử lý như audio...")
        return input_path


def save_benchmark(audio_path, benchmark_data):
    """Lưu benchmark log để so sánh performance."""
    # ✅ FIX: Lưu vào data/logs/benchmarks
    benchmark_dir = os.path.join("data", "logs", "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(audio_path).split('.')[0]
    benchmark_path = os.path.join(benchmark_dir, f"{filename}_{timestamp}.json")
    
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Benchmark saved: {benchmark_path}")
    return benchmark_path


def estimate_remaining_time(step_name, elapsed, total_duration_minutes):
    """Ước tính thời gian còn lại dựa trên thống kê."""
    # Tỷ lệ trung bình từ thống kê thực tế
    time_ratios = {
        "preprocessing": 0.1,     # ~10% tổng thời gian
        "diarization": 0.35,      # ~35% (chậm nhất)
        "transcription": 0.45,    # ~45% 
        "gender": 0.05,           # ~5%
        "combine": 0.05,          # ~5%
    }
    
    estimated_total = 0
    completed = 0
    
    if step_name == "preprocessing":
        # Dự đoán tổng thời gian dựa trên preprocessing
        estimated_total = elapsed / time_ratios["preprocessing"]
        completed = time_ratios["preprocessing"]
    elif step_name == "diarization":
        estimated_total = elapsed / (time_ratios["preprocessing"] + time_ratios["diarization"])
        completed = time_ratios["preprocessing"] + time_ratios["diarization"]
    elif step_name == "transcription":
        estimated_total = elapsed / (time_ratios["preprocessing"] + time_ratios["diarization"] + time_ratios["transcription"])
        completed = time_ratios["preprocessing"] + time_ratios["diarization"] + time_ratios["transcription"]
    elif step_name == "gender":
        estimated_total = elapsed / (1 - time_ratios["combine"])
        completed = 1 - time_ratios["combine"]
    else:
        estimated_total = elapsed
        completed = 1.0
    
    remaining = estimated_total * (1 - completed)
    return remaining, completed * 100


def main(audio_path, num_speakers=None, min_speakers=None, max_speakers=None, 
         chunk_minutes=10, use_parallel=False, use_cache=True, force_rerun=False):
    """
    Optimized pipeline với caching.
    
    Args:
        use_parallel: Sử dụng parallel transcription (nhanh hơn)
        use_cache: Sử dụng cache (nhanh hơn nhiều nếu đã chạy trước đó)
        force_rerun: Xóa cache và chạy lại từ đầu
    """
    total_start = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ✅ Thu thập thông tin hệ thống
    import soundfile as sf
    import platform
    import torch
    
    audio_info = sf.info(audio_path)
    duration_minutes = audio_info.duration / 60
    
    system_info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "audio_duration_minutes": duration_minutes,
        "sample_rate": audio_info.samplerate,
        "channels": audio_info.channels,
    }
    
    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZED PIPELINE")
    print(f"📅 Started: {start_datetime}")
    print(f"📁 Input: {audio_path}")
    print(f"⚡ Cache: {use_cache}, Parallel: {use_parallel}")
    print(f"💻 Device: {system_info['device']}")
    print(f"⏱️  Duration: {duration_minutes:.1f} minutes")
    print(f"{'='*60}\n")
    
    # Clear cache if force_rerun
    if force_rerun and use_cache:
        print("🗑️  Clearing cache...")
        clear_cache(audio_path)
    
    # Convert video if needed
    from pipeline_optimized import convert_to_audio
    audio_path = convert_to_audio(audio_path)
    
    # ✅ Benchmark data
    benchmark = {
        "timestamp": start_datetime,
        "system_info": system_info,
        "config": {
            "use_cache": use_cache,
            "use_parallel": use_parallel,
            "chunk_minutes": chunk_minutes,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
        "steps": {}
    }
    
    # Step 0: Audio preprocessing
    base_name = os.path.splitext(audio_path)[0]
    enhanced_path = base_name + '_enhanced.wav'
    
    if use_cache and os.path.exists(enhanced_path):
        print("💾 Using cached enhanced audio")
        audio_path = enhanced_path
        step0_time = 0
    else:
        print("🔧 Step 0: Audio Preprocessing...")
        step0_start = time.time()
        audio_path = enhance_audio(audio_path, enhanced_path)
        step0_time = time.time() - step0_start
        
        # ✅ Hiển thị ETA
        eta, progress = estimate_remaining_time("preprocessing", step0_time, duration_minutes)
        print(f"⏱️  Step 0: {step0_time:.2f}s")
        print(f"📊 Progress: {progress:.1f}% | ETA: {eta/60:.1f} minutes remaining\n")
    
    benchmark["steps"]["preprocessing"] = {
        "time_seconds": step0_time,
        "cached": step0_time == 0
    }
    
    paths = prepare_output_paths(audio_path)
    
    # Step 1: Diarization (with cache)
    step1_start = time.time()
    diar_cache = load_cache(audio_path, "diarization") if use_cache else None
    
    if diar_cache and os.path.exists(paths["diarization"]):
        print("💾 Using cached diarization")
        step1_time = 0
    else:
        print("🔊 Step 1: Speaker Diarization...")
        diarize_audio(
            audio_path, 
            paths["diarization"],
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        step1_time = time.time() - step1_start
        
        if use_cache:
            save_cache(audio_path, "diarization", {"path": paths["diarization"]})
    
    # ✅ Hiển thị ETA sau diarization
    total_elapsed = time.time() - total_start
    eta, progress = estimate_remaining_time("diarization", total_elapsed, duration_minutes)
    print(f"⏱️  Step 1: {step1_time:.2f}s")
    print(f"📊 Progress: {progress:.1f}% | ETA: {eta/60:.1f} minutes remaining\n")
    
    benchmark["steps"]["diarization"] = {
        "time_seconds": step1_time,
        "cached": step1_time == 0
    }
    
    # Step 2: Transcription (with cache & parallel option)
    step2_start = time.time()
    trans_cache = load_cache(audio_path, "transcription") if use_cache else None
    
    if trans_cache and os.path.exists(paths["transcription"]):
        print("💾 Using cached transcription")
        step2_time = 0
    else:
        print("🎧 Step 2: Speech-to-Text...")
        
        if use_parallel and HAS_PARALLEL:
            print("   ⚡ Using PARALLEL transcription")
            transcribe_audio_parallel(
                audio_path, 
                paths["transcription"],
                chunk_length_minutes=chunk_minutes,
                diarization_path=paths["diarization"]
            )
        elif HAS_OPTIMIZED:
            print("   ⚡ Using OPTIMIZED transcription (faster-whisper)")
            transcribe_audio_optimized(
                audio_path, 
                paths["transcription"],
                chunk_length_minutes=chunk_minutes,
                diarization_path=paths["diarization"]
            )
        else:
            print("   ⚠️  Using STANDARD transcription")
            from transcribe import transcribe_audio
            transcribe_audio(
                audio_path, 
                paths["transcription"],
                chunk_length_minutes=chunk_minutes,
                diarization_path=paths["diarization"]
            )
        
        step2_time = time.time() - step2_start
        
        if use_cache:
            save_cache(audio_path, "transcription", {"path": paths["transcription"]})
    
    # ✅ Hiển thị ETA sau transcription
    total_elapsed = time.time() - total_start
    eta, progress = estimate_remaining_time("transcription", total_elapsed, duration_minutes)
    print(f"⏱️  Step 2: {step2_time:.2f}s")
    print(f"📊 Progress: {progress:.1f}% | ETA: {eta/60:.1f} minutes remaining\n")
    
    benchmark["steps"]["transcription"] = {
        "time_seconds": step2_time,
        "cached": step2_time == 0,
        "method": "parallel" if use_parallel and HAS_PARALLEL else ("optimized" if HAS_OPTIMIZED else "standard")
    }
    
    # Step 3: Gender Classification (with cache)
    step3_start = time.time()
    gender_cache = load_cache(audio_path, "gender") if use_cache else None
    
    if gender_cache and os.path.exists(paths["gender"]):
        print("💾 Using cached gender classification")
        step3_time = 0
    else:
        print("👤 Step 3: Gender Classification...")
        classify_gender(paths["diarization"], paths["gender"], audio_path=audio_path)
        step3_time = time.time() - step3_start
        
        if use_cache:
            save_cache(audio_path, "gender", {"path": paths["gender"]})
    
    print(f"⏱️  Step 3: {step3_time:.2f}s\n")
    
    benchmark["steps"]["gender_classification"] = {
        "time_seconds": step3_time,
        "cached": step3_time == 0
    }
    
    # Step 4: Combine
    step4_start = time.time()
    print("🔗 Step 4: Combining Results...")
    combine_results(paths["diarization"], paths["transcription"], paths["gender"], paths["combined"])
    step4_time = time.time() - step4_start
    print(f"⏱️  Step 4: {step4_time:.2f}s\n")
    
    benchmark["steps"]["combine"] = {
        "time_seconds": step4_time
    }
    
    total_elapsed = time.time() - total_start
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ✅ Tính toán metrics
    actual_steps_time = step0_time + step1_time + step2_time + step3_time + step4_time
    cache_saved_time = total_elapsed - actual_steps_time if actual_steps_time < total_elapsed * 0.5 else 0
    speedup = duration_minutes / (total_elapsed / 60) if total_elapsed > 0 else 0
    
    benchmark["summary"] = {
        "total_time_seconds": total_elapsed,
        "actual_processing_time": actual_steps_time,
        "cache_saved_time": cache_saved_time,
        "speedup_vs_realtime": speedup,
        "end_datetime": end_datetime
    }
    
    print(f"\n{'='*60}")
    print("🎉 OPTIMIZED PIPELINE COMPLETED")
    print(f"📅 Finished: {end_datetime}")
    print(f"📂 Results: {paths['output_dir']}")
    print(f"\n📊 STATISTICS:")
    print(f"   0️⃣  Preprocessing:    {step0_time:>8.2f}s")
    print(f"   1️⃣  Diarization:      {step1_time:>8.2f}s")
    print(f"   2️⃣  Transcription:    {step2_time:>8.2f}s")
    print(f"   3️⃣  Gender classify:  {step3_time:>8.2f}s")
    print(f"   4️⃣  Combine:          {step4_time:>8.2f}s")
    print(f"   {'─'*50}")
    print(f"   🕓 TOTAL:             {total_elapsed:>8.2f}s ({total_elapsed/60:.1f} min)")
