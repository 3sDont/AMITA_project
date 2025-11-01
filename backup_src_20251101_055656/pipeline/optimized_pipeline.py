"""
Pipeline tối ưu với:
- Caching (tránh chạy lại)
- Parallel transcription
- Progress tracking
- Error recovery
"""
import sys
import time
import os
from datetime import datetime
import argparse

from utils import prepare_output_paths
from diarize import diarize_audio
from transcribe_optimized import transcribe_audio_optimized
from transcribe_parallel import transcribe_audio_parallel
from gender_classify import classify_gender
from combine import combine_results
from audio_preprocess import enhance_audio
from cache_manager import load_cache, save_cache, clear_cache


def run_pipeline_optimized(
    audio_path,
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
    chunk_minutes=10,
    use_parallel=False,
    use_cache=True,
    force_rerun=False
):
    """
    Pipeline tối ưu với caching và parallel processing.
    
    Args:
        use_parallel: Sử dụng parallel transcription (nhanh hơn trên CPU nhiều cores)
        use_cache: Sử dụng cache (tránh chạy lại các bước đã hoàn thành)
        force_rerun: Bỏ qua cache, chạy lại toàn bộ
    """
    total_start = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU PIPELINE (OPTIMIZED)")
    print(f"📅 Thời gian bắt đầu: {start_datetime}")
    print(f"📁 File input: {audio_path}")
    print(f"⚡ Parallel: {use_parallel}, Cache: {use_cache}")
    print(f"{'='*60}\n")
    
    # Xóa cache nếu force_rerun
    if force_rerun and use_cache:
        clear_cache(audio_path)
    
    # Convert & enhance audio
    from pipeline import convert_to_audio
    audio_path = convert_to_audio(audio_path)
    
    print("🔧 Tiền xử lý audio...")
    base_name = os.path.splitext(audio_path)[0]
    enhanced_path = base_name + '_enhanced.wav'
    
    # ✅ Kiểm tra cache cho enhanced audio
    if use_cache and os.path.exists(enhanced_path):
        print("   💾 Tìm thấy enhanced audio cached")
        audio_path = enhanced_path
    else:
        audio_path = enhance_audio(audio_path, enhanced_path)
    
    paths = prepare_output_paths(audio_path)
    
    # 1️⃣ Diarization
    step1_start = time.time()
    diar_cache = load_cache(audio_path, "diarization") if use_cache else None
    
    if diar_cache and os.path.exists(paths["diarization"]):
        print("💾 Sử dụng diarization từ cache")
        step1_time = 0
    else:
        diarize_audio(audio_path, paths["diarization"], num_speakers, min_speakers, max_speakers)
        step1_time = time.time() - step1_start
        if use_cache:
            save_cache(audio_path, "diarization", {"path": paths["diarization"]})
    
    print(f"⏱️  Bước 1: {step1_time:.2f}s\n")
    
    # 2️⃣ Transcription
    step2_start = time.time()
    trans_cache = load_cache(audio_path, "transcription") if use_cache else None
    
    if trans_cache and os.path.exists(paths["transcription"]):
        print("💾 Sử dụng transcription từ cache")
        step2_time = 0
    else:
        if use_parallel:
            transcribe_audio_parallel(audio_path, paths["transcription"], chunk_minutes, paths["diarization"])
        else:
            transcribe_audio_optimized(audio_path, paths["transcription"], chunk_minutes, paths["diarization"])
        step2_time = time.time() - step2_start
        if use_cache:
            save_cache(audio_path, "transcription", {"path": paths["transcription"]})
    
    print(f"⏱️  Bước 2: {step2_time:.2f}s\n")
    
    # 3️⃣ Gender Classification
    step3_start = time.time()
    gender_cache = load_cache(audio_path, "gender") if use_cache else None
    
    if gender_cache and os.path.exists(paths["gender"]):
        print("💾 Sử dụng gender classification từ cache")
        step3_time = 0
    else:
        classify_gender(paths["diarization"], paths["gender"], audio_path=audio_path)
        step3_time = time.time() - step3_start
        if use_cache:
            save_cache(audio_path, "gender", {"path": paths["gender"]})
    
    print(f"⏱️  Bước 3: {step3_time:.2f}s\n")
    
    # 4️⃣ Combine
    step4_start = time.time()
    combine_results(paths["diarization"], paths["transcription"], paths["gender"], paths["combined"])
    step4_time = time.time() - step4_start
    print(f"⏱️  Bước 4: {step4_time:.2f}s\n")
    
    total_elapsed = time.time() - total_start
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print("🎉 HOÀN TẤT PIPELINE (OPTIMIZED)")
    print(f"📅 Kết thúc: {end_datetime}")
    print(f"📂 Kết quả: {paths['output_dir']}")
    print(f"\n📊 THỐNG KÊ:")
    print(f"   1️⃣  Diarization:     {step1_time:>8.2f}s")
    print(f"   2️⃣  Transcription:   {step2_time:>8.2f}s")
    print(f"   3️⃣  Gender classify: {step3_time:>8.2f}s")
    print(f"   4️⃣  Combine:         {step4_time:>8.2f}s")
    print(f"   {'─'*50}")
    print(f"   🕓 TỔNG:             {total_elapsed:>8.2f}s ({total_elapsed/60:>5.1f} phút)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized audio processing pipeline")
    parser.add_argument("audio_path", type=str)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--chunk-minutes", type=int, default=10)
    parser.add_argument("--parallel", action="store_true", help="Sử dụng parallel transcription")
    parser.add_argument("--no-cache", action="store_true", help="Tắt caching")
    parser.add_argument("--force-rerun", action="store_true", help="Bỏ qua cache, chạy lại toàn bộ")
    
    args = parser.parse_args()
    
    run_pipeline_optimized(
        args.audio_path,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        chunk_minutes=args.chunk_minutes,
        use_parallel=args.parallel,
        use_cache=not args.no_cache,
        force_rerun=args.force_rerun
    )
