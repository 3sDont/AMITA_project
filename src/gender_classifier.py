import sys
import os
# Thêm thư mục gốc vào sys.path để import config và utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import json
import time
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import config
import utils


def extract_pitch_features(audio_segment, sr=16000):
    """
    Trích xuất đặc trưng pitch (tần số cơ bản) để phân biệt nam/nữ.
    Nam: 85-180 Hz
    Nữ: 165-255 Hz
    """
    # Sử dụng librosa để tính pitch
    pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr, fmin=50, fmax=400)

    # Lấy pitch values có magnitude đủ lớn
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Chỉ lấy pitch hợp lệ
            pitch_values.append(pitch)

    if len(pitch_values) == 0:
        return None

    return {
        'mean': np.mean(pitch_values),
        'median': np.median(pitch_values),
        'std': np.std(pitch_values),
        'min': np.min(pitch_values),
        'max': np.max(pitch_values)
    }


def classify_gender_from_pitch(pitch_features):
    """
    Phân loại giới tính dựa trên pitch.

    Ngưỡng:
    - pitch_mean < 165 Hz → Male
    - pitch_mean > 190 Hz → Female
    - 165-190 Hz → Dựa vào median và std
    """
    if pitch_features is None:
        return "Unknown"

    mean_pitch = pitch_features['mean']
    median_pitch = pitch_features['median']

    # Ngưỡng rõ ràng
    if mean_pitch < 155:
        return "Male"
    elif mean_pitch > 200:
        return "Female"

    # Vùng trung gian: dùng median để quyết định
    if 155 <= mean_pitch <= 200:
        if median_pitch < 175:
            return "Male"
        else:
            return "Female"

    return "Unknown"


def extract_speaker_audio(audio_path, diarization_segments, speaker_id, max_duration=30):
    """
    Trích xuất audio của 1 speaker cụ thể từ file gốc.
    Chỉ lấy tối đa max_duration giây để tăng tốc.
    """
    # Load toàn bộ audio
    waveform, sr = sf.read(audio_path)

    # Nếu stereo → mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Lọc các segments của speaker này
    speaker_segments = [seg for seg in diarization_segments if seg['speaker'] == speaker_id]

    if not speaker_segments:
        return None, sr

    # Ghép các đoạn audio lại (tối đa max_duration giây)
    collected_audio = []
    total_duration = 0

    for seg in speaker_segments:
        if total_duration >= max_duration:
            break

        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        segment_audio = waveform[start_sample:end_sample]

        collected_audio.append(segment_audio)
        total_duration += (end_sample - start_sample) / sr

    if not collected_audio:
        return None, sr

    return np.concatenate(collected_audio), sr


def classify_gender(audio_path=None):
    """
    Phân loại giới tính người nói dựa trên phân tích pitch.
    """
    print("👤 Phân loại giới tính...")
    #start_time = time.time()
    #start_dt = datetime.now().strftime("%H:%M:%S")
    #print(f"   ⏰ Bắt đầu lúc: {start_dt}")

    # ✅ Load diarization segments từ JSON
    if not os.path.exists(config.DIARIZATION_CACHE):
        print(f"❌ Lỗi: Không tìm thấy file diarization: {config.DIARIZATION_CACHE}")
        print(f"💡 Vui lòng chạy src/diarization.py trước!")
        return {}
    
    print(f"   📂 Load diarization từ: {config.DIARIZATION_CACHE}")
    diar_data = utils.load_json(config.DIARIZATION_CACHE)
    
    # ✅ Xử lý cả 2 format: list hoặc dict với key 'segments'
    if isinstance(diar_data, list):
        diar_segments = diar_data
    else:
        diar_segments = diar_data.get('segments', [])
    
    # Lấy danh sách speakers từ segments
    speakers = set(seg['speaker'] for seg in diar_segments)
    
    print(f"   👥 Phát hiện {len(speakers)} người nói")
    print(f"   ⏳ Ước tính: ~{len(speakers) * 0.5:.1f}s\n")

    genders = {}

    # Nếu có audio gốc, phân tích thật
    if audio_path and os.path.exists(audio_path):
        print(f"   🎵 Phân tích đặc trưng âm thanh từ: {os.path.basename(audio_path)}")
        
        # ✅ Load audio một lần
        #load_start = time.time()
        import soundfile as sf
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        #load_time = time.time() - load_start
        #print(f"   ✅ Load audio: {load_time:.1f}s\n")

        # Sử dụng diar_segments đã load ở trên
        for idx, speaker in enumerate(sorted(speakers), 1):
            #speaker_start = time.time()
            print(f"   🎤 Xử lý {speaker} ({idx}/{len(speakers)})...")

            # Trích xuất audio của speaker
            #extract_start = time.time()
            
            # ✅ Tối ưu: Sử dụng waveform đã load
            speaker_segments = [seg for seg in diar_segments if seg['speaker'] == speaker]
            
            if not speaker_segments:
                print(f"      ⚠️  Không có segments → Unknown")
                genders[speaker] = "Unknown"
                continue
            
            collected_audio = []
            total_duration = 0
            max_duration = 30
            
            for seg in speaker_segments:
                if total_duration >= max_duration:
                    break
                start_sample = int(seg['start_time'] * sr)  # ✅ Sử dụng start_time
                end_sample = int(seg['end_time'] * sr)      # ✅ Sử dụng end_time
                segment_audio = waveform[start_sample:end_sample]
                collected_audio.append(segment_audio)
                total_duration += (end_sample - start_sample) / sr
            
            if not collected_audio:
                genders[speaker] = "Unknown"
                continue
            
            speaker_audio = np.concatenate(collected_audio)
            #extract_time = time.time() - extract_start
            
            if len(speaker_audio) < sr * 0.5:
                print(f"      ⚠️  Không đủ audio ({len(speaker_audio)/sr:.1f}s) → Unknown")
                genders[speaker] = "Unknown"
                continue

            # ✅ Trích xuất pitch features
            #pitch_start = time.time()
            pitch_features = extract_pitch_features(speaker_audio, sr)
            #pitch_time = time.time() - pitch_start

            if pitch_features:
                print(f"      📊 Pitch mean: {pitch_features['mean']:.1f} Hz")
                print(f"      📊 Pitch median: {pitch_features['median']:.1f} Hz")

            # Phân loại
            gender = classify_gender_from_pitch(pitch_features)
            genders[speaker] = gender
            
            #speaker_time = time.time() - speaker_start
            #print(f"      ✅ Kết quả: {gender} ({speaker_time:.1f}s)")
            #print(f"         [Extract: {extract_time:.1f}s, Pitch: {pitch_time:.1f}s]\n")

    else:
        # Fallback
        print("   ⚠️  Không có audio gốc, sử dụng heuristic...")
        for i, speaker in enumerate(sorted(speakers)):
            genders[speaker] = "Male" if i % 2 == 0 else "Female"

    # ✅ Lưu kết quả
    print("   💾 Đang lưu kết quả...")
    utils.save_json(genders, config.GENDER_CACHE)
    
    #elapsed = time.time() - start_time
    #end_dt = datetime.now().strftime("%H:%M:%S")
    #print(f"⏱️  Thời gian gender classification: {elapsed:.2f}s")
    #print(f"   ⏰ Kết thúc lúc: {end_dt}")

    # In tóm tắt
    print("\n   📊 Tóm tắt:")
    for speaker, gender in sorted(genders.items()):
        print(f"      {speaker}: {gender}")

    return genders


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("👤 GIAI ĐOẠN 2.5: GENDER CLASSIFICATION")
        print("="*80 + "\n")
        
        # Kiểm tra file audio có tồn tại không
        if not os.path.exists(config.AUDIO_FILE):
            print(f"❌ Lỗi: Không tìm thấy file audio: {config.AUDIO_FILE}")
            print(f"💡 Vui lòng kiểm tra lại đường dẫn trong config.py")
            sys.exit(1)
        
        # Kiểm tra diarization cache có tồn tại không
        if not os.path.exists(config.DIARIZATION_CACHE):
            print(f"❌ Lỗi: Không tìm thấy file diarization: {config.DIARIZATION_CACHE}")
            print(f"💡 Vui lòng chạy src/diarization.py trước!")
            sys.exit(1)
        
        # Tạo thư mục outputs nếu chưa có
        os.makedirs("outputs", exist_ok=True)
        
        print(f"📁 Input Audio: {config.AUDIO_FILE}")
        print(f"📁 Input Diarization: {config.DIARIZATION_CACHE}")
        print(f"⚙️  Config: USE_GPU={config.USE_GPU}\n")
        
        # Chạy gender classification
        result = classify_gender(audio_path=config.AUDIO_FILE)
        
        print(f"\n✓ Kết quả đã lưu tại: {config.GENDER_CACHE}")
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH GIAI ĐOẠN 2.5")
        print("="*80)
        print(f"👥 Số người nói: {len(result)}")
        
        # Hiển thị kết quả
        print(f"\n📋 Kết quả gender classification:")
        print("-" * 80)
        male_count = sum(1 for g in result.values() if g == "Male")
        female_count = sum(1 for g in result.values() if g == "Female")
        unknown_count = sum(1 for g in result.values() if g == "Unknown")
        
        for speaker, gender in sorted(result.items()):
            emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
            print(f"   {emoji} {speaker}: {gender}")
        
        print("\n📊 Thống kê:")
        print(f"   👨 Nam: {male_count}")
        print(f"   👩 Nữ: {female_count}")
        if unknown_count > 0:
            print(f"   ❓ Không xác định: {unknown_count}")
        
        print("\n" + "="*80)
        print("💡 TIP: Chạy tiếp giai đoạn 3 (mapping) nếu chưa có:")
        print("   python src/step3_mapping.py")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
