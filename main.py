"""
FILE CHÍNH - Pipeline tối ưu cho audio dài (>1 giờ)
"""
import sys
import os
# Thêm thư mục gốc và src vào sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import warnings
warnings.filterwarnings('ignore')  # Tắt warnings

import time
from datetime import datetime
import config
import utils

# Import các modules từ src/
from src import audio_processor
from src import whisper
from src import diarization
from src import gender_classifier
from src import combiner


def create_final_output(combined_segments):
    """Tạo output hoàn chỉnh từ combining results"""
    # Load combining data
    data = utils.load_json(config.COMBINING_CACHE)
    
    # Tính statistics
    speakers = {}
    total_duration = 0
    
    for seg in data:
        speaker = seg['speaker']
        duration = seg['end'] - seg['start']
        
        if speaker not in speakers:
            speakers[speaker] = {
                'segments': 0,
                'duration': 0,
                'gender': seg.get('gender', 'Unknown')
            }
        
        speakers[speaker]['segments'] += 1
        speakers[speaker]['duration'] += duration
        total_duration += duration
    
    return {
        "metadata": {
            "audio_file": config.AUDIO_FILE,
            "processed_at": datetime.now().isoformat(),
            "total_duration": total_duration,
            "number_of_speakers": len(speakers),
            "number_of_segments": len(data),
            "whisper_model": config.WHISPER_MODEL,
            "language": config.LANGUAGE,
            "use_gpu": config.USE_GPU,
            "llm_enabled": config.ENABLE_LLM_ANALYSIS,
            "llm_model": config.OLLAMA_MODEL if config.ENABLE_LLM_ANALYSIS else None
        },
        "speaker_statistics": speakers,
        "transcript": data
    }


def main():
    """Chạy toàn bộ pipeline - OPTIMIZED ORDER"""
    print("\n" + "="*80)
    print("🎯 OPTIMIZED PIPELINE: PREPROCESSING → DIARIZATION → GENDER → WHISPER → COMBINING → LLM")
    print("="*80 + "\n")
    
    print(f"📁 File audio gốc: {config.AUDIO_FILE}")
    print(f"📄 Output: {config.OUTPUT_JSON}")
    print(f"🎤 Whisper model: {config.WHISPER_MODEL}")
    print(f"🗣️  Ngôn ngữ: {config.LANGUAGE}")
    print(f"🎮 GPU: {'Enabled' if config.USE_GPU else 'Disabled'}")
    print(f"🤖 LLM: {'Ollama (' + config.OLLAMA_MODEL + ')' if config.ENABLE_LLM_ANALYSIS else 'Disabled'}")
    
    start_time = time.time()
    stage_times = {}
    
    try:
        # ===== GIAI ĐOẠN 0: Audio Preprocessing =====
        print("\n" + "="*80)
        print("🎵 GIAI ĐOẠN 0: AUDIO PREPROCESSING")
        print("="*80)
        stage_start = time.time()
        
        basename = os.path.splitext(os.path.basename(config.AUDIO_FILE))[0]
        enhanced_audio_path = f"outputs/{basename}_enhanced.wav"
        
        processed_audio = audio_processor.enhance_audio(
            input_path=config.AUDIO_FILE,
            output_path=enhanced_audio_path
        )
        
        stage_times['Audio Preprocessing'] = time.time() - stage_start
        print(f"\n⏱️  Giai đoạn 0 hoàn thành: {stage_times['Audio Preprocessing']:.2f}s ({stage_times['Audio Preprocessing']/60:.2f} phút)")
        
        # ===== GIAI ĐOẠN 1: Speaker Diarization (ƯU TIÊN) =====
        print("\n" + "="*80)
        print("👥 GIAI ĐOẠN 1: SPEAKER DIARIZATION (PRIORITY)")
        print("="*80)
        print("💡 Chạy trước để biết ai nói khi nào, giúp Whisper xử lý hiệu quả hơn")
        stage_start = time.time()
        
        diarization_result = diarization.diarize_audio(
            audio_path=processed_audio,
            num_speakers=None,
            min_speakers=config.MIN_SPEAKERS,
            max_speakers=config.MAX_SPEAKERS
        )
        
        # ✅ LƯU KẾT QUẢ DIARIZATION NGAY LẬP TỨC
        utils.save_json(diarization_result, config.DIARIZATION_CACHE)
        print(f"✓ Đã lưu: {config.DIARIZATION_CACHE}")
        
        stage_times['Speaker Diarization'] = time.time() - stage_start
        print(f"\n⏱️  Giai đoạn 1 hoàn thành: {stage_times['Speaker Diarization']:.2f}s ({stage_times['Speaker Diarization']/60:.2f} phút)")
        
        # ===== GIAI ĐOẠN 2: Gender Classification =====
        print("\n" + "="*80)
        print("👤 GIAI ĐOẠN 2: GENDER CLASSIFICATION")
        print("="*80)
        stage_start = time.time()
        
        gender_result = gender_classifier.classify_gender(audio_path=processed_audio)
        
        # ✅ LƯU KẾT QUẢ GENDER NGAY LẬP TỨC (đã có trong function nhưng đảm bảo)
        utils.save_json(gender_result, config.GENDER_CACHE)
        print(f"✓ Đã lưu: {config.GENDER_CACHE}")
        
        stage_times['Gender Classification'] = time.time() - stage_start
        print(f"\n⏱️  Giai đoạn 2 hoàn thành: {stage_times['Gender Classification']:.2f}s ({stage_times['Gender Classification']/60:.2f} phút)")
        
        # ===== GIAI ĐOẠN 3: WHISPER TRANSCRIPTION (SAU DIARIZATION) =====
        print("\n" + "="*80)
        print("🎤 GIAI ĐOẠN 3: WHISPER TRANSCRIPTION (với diarization info)")
        print("="*80)
        print("💡 Sử dụng thông tin diarization để chia chunks thông minh hơn")
        stage_start = time.time()
        
        # ✅ Whisper giờ có thể dùng diarization để chia chunks tốt hơn
        whisper_result = whisper.transcribe_audio_optimized(
            audio_path=processed_audio,
            chunk_length_minutes=10  # Có thể tăng lên 15-20 cho file dài
        )
        
        stage_times['Whisper Transcription'] = time.time() - stage_start
        print(f"\n⏱️  Giai đoạn 3 hoàn thành: {stage_times['Whisper Transcription']:.2f}s ({stage_times['Whisper Transcription']/60:.2f} phút)")
        
        # ===== GIAI ĐOẠN 4: Combining Results =====
        print("\n" + "="*80)
        print("🔗 GIAI ĐOẠN 4: COMBINING RESULTS")
        print("="*80)
        stage_start = time.time()
        
        combined_result = combiner.combine_results()
        
        stage_times['Combining'] = time.time() - stage_start
        print(f"\n⏱️  Giai đoạn 4 hoàn thành: {stage_times['Combining']:.2f}s ({stage_times['Combining']/60:.2f} phút)")
        
        # ===== GIAI ĐOẠN 5: LLM Analysis (optional) =====
        if config.ENABLE_LLM_ANALYSIS:
            print("\n" + "="*80)
            print("🧠 GIAI ĐOẠN 5: LLM ANALYSIS")
            print("="*80)
            print("\n💡 Để chạy LLM analysis, sử dụng:")
            print("   python src/llm_applying.py")
            print("\n⚠️  Bỏ qua LLM analysis trong main pipeline")
        
        # ===== Tạo Final Output =====
        print("\n" + "="*80)
        print("💾 TẠO FINAL OUTPUT")
        print("="*80)
        
        final_output = create_final_output(combined_result)
        utils.save_json(final_output, config.OUTPUT_JSON)
        print(f"✓ Đã lưu final output: {config.OUTPUT_JSON}")
        
        # ===== Preview & Stats (giữ nguyên) =====
        print("\n" + "="*80)
        print("📊 THỐNG KÊ & PREVIEW")
        print("="*80)
        
        # Load combining data để preview
        combined_data = utils.load_json(config.COMBINING_CACHE)
        
        print(f"\n📈 Tổng quan:")
        print(f"   • Tổng segments: {len(combined_data)}")
        print(f"   • Số người nói: {final_output['metadata']['number_of_speakers']}")
        print(f"   • Tổng thời gian: {final_output['metadata']['total_duration']:.1f}s")
        
        print(f"\n👥 Thống kê theo người nói:")
        for speaker, stats in final_output['speaker_statistics'].items():
            gender_emoji = "👨" if stats['gender'] == "Male" else "👩" if stats['gender'] == "Female" else "❓"
            print(f"   {gender_emoji} {speaker} ({stats['gender']}):")
            print(f"      - Segments: {stats['segments']}")
            print(f"      - Duration: {stats['duration']:.1f}s")
        
        print(f"\n📄 Preview 3 segments đầu tiên:")
        print("-" * 80)
        for i, seg in enumerate(combined_data[:3]):
            duration = seg['end'] - seg['start']
            gender_emoji = "👨" if seg.get('gender') == "Male" else "👩" if seg.get('gender') == "Female" else "❓"
            print(f"\n[{i+1}] {seg['start']:.2f}s → {seg['end']:.2f}s (dài {duration:.1f}s)")
            print(f"    {gender_emoji} {seg['speaker']} ({seg.get('gender', 'Unknown')})")
            print(f"    Text: {seg['text'][:100]}{'...' if len(seg['text']) > 100 else ''}")
        
        if len(combined_data) > 3:
            print(f"\n... và {len(combined_data) - 3} segments khác")
        
        # ===== Timing Summary =====
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("⏱️  THỜI GIAN XỬ LÝ TỪNG GIAI ĐOẠN")
        print("="*80)
        
        for stage_name, stage_time in stage_times.items():
            percentage = (stage_time / elapsed_time) * 100
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"{stage_name:25s} │ {bar} │ {stage_time:7.2f}s ({stage_time/60:5.2f}m) - {percentage:5.1f}%")
        
        print("-"*80)
        print(f"{'TỔNG THỜI GIAN':25s} │ {'█' * 50} │ {elapsed_time:7.2f}s ({elapsed_time/60:5.2f}m) - 100.0%")
        
        # ===== Final Summary =====
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print("="*80)
        
        print(f"\n📁 Các file đã tạo:")
        print(f"   1. {config.WHISPER_CACHE:50s} - Whisper transcription")
        print(f"   2. {config.DIARIZATION_CACHE:50s} - Speaker diarization")
        print(f"   3. {config.GENDER_CACHE:50s} - Gender classification")
        print(f"   4. {config.COMBINING_CACHE:50s} - Combined results")
        print(f"   5. {config.OUTPUT_JSON:50s} - Final output")
        
        if config.ENABLE_LLM_ANALYSIS:
            print(f"\n💡 Để chạy LLM analysis:")
            print(f"   python src/step4_llm_analysis.py")
            print(f"   → Tạo: outputs/dialog.txt, meeting_summary.txt, meeting_tasks.json")
        
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
