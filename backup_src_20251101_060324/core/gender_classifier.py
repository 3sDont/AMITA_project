import json
import time
import os
from datetime import datetime
import numpy as np
import librosa
import soundfile as sf
from scipy import signal


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


def load_diarization_segments(diarization_path):
    """Load diarization segments từ file txt."""
    import re
    segments = []
    time_pattern = re.compile(r"([\d.]+)s\s+–\s+([\d.]+)s\s+:\s+(SPEAKER_\d+)")

    with open(diarization_path, "r", encoding="utf-8") as f:
        for line in f:
            match = time_pattern.search(line)
            if match:
                start, end, speaker = match.groups()
                segments.append({
                    "start": float(start),
                    "end": float(end),
                    "speaker": speaker
                })
    return segments


def classify_gender(diarization_path, output_path, audio_path=None):
    """
    Phân loại giới tính người nói dựa trên phân tích pitch.
    """
    print("👤 Phân loại giới tính...")
    start_time = time.time()
    start_dt = datetime.now().strftime("%H:%M:%S")
    print(f"   ⏰ Bắt đầu lúc: {start_dt}")

    # Đọc danh sách speakers từ diarization
    speakers = set()
    with open(diarization_path, "r", encoding="utf-8") as f:
        for line in f:
            if "SPEAKER_" in line:
                spk = line.split(":")[-1].strip()
                speakers.add(spk)
    
    print(f"   👥 Phát hiện {len(speakers)} người nói")
    print(f"   ⏳ Ước tính: ~{len(speakers) * 0.5:.1f}s\n")

    genders = {}

    # Nếu có audio gốc, phân tích thật
    if audio_path and os.path.exists(audio_path):
        print(f"   🎵 Phân tích đặc trưng âm thanh từ: {os.path.basename(audio_path)}")
        
        # ✅ Load audio một lần
        load_start = time.time()
        import soundfile as sf
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        load_time = time.time() - load_start
        print(f"   ✅ Load audio: {load_time:.1f}s\n")

        # Load diarization segments
        diar_segments = load_diarization_segments(diarization_path)

        for idx, speaker in enumerate(sorted(speakers), 1):
            speaker_start = time.time()
            print(f"   🎤 Xử lý {speaker} ({idx}/{len(speakers)})...")

            # Trích xuất audio của speaker
            extract_start = time.time()
            
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
                start_sample = int(seg['start'] * sr)
                end_sample = int(seg['end'] * sr)
                segment_audio = waveform[start_sample:end_sample]
                collected_audio.append(segment_audio)
                total_duration += (end_sample - start_sample) / sr
            
            if not collected_audio:
                genders[speaker] = "Unknown"
                continue
            
            speaker_audio = np.concatenate(collected_audio)
            extract_time = time.time() - extract_start
            
            if len(speaker_audio) < sr * 0.5:
                print(f"      ⚠️  Không đủ audio ({len(speaker_audio)/sr:.1f}s) → Unknown")
                genders[speaker] = "Unknown"
                continue

            # ✅ Trích xuất pitch features
            pitch_start = time.time()
            pitch_features = extract_pitch_features(speaker_audio, sr)
            pitch_time = time.time() - pitch_start

            if pitch_features:
                print(f"      📊 Pitch mean: {pitch_features['mean']:.1f} Hz")
                print(f"      📊 Pitch median: {pitch_features['median']:.1f} Hz")

            # Phân loại
            gender = classify_gender_from_pitch(pitch_features)
            genders[speaker] = gender
            
            speaker_time = time.time() - speaker_start
            print(f"      ✅ Kết quả: {gender} ({speaker_time:.1f}s)")
            print(f"         [Extract: {extract_time:.1f}s, Pitch: {pitch_time:.1f}s]\n")

    else:
        # Fallback
        print("   ⚠️  Không có audio gốc, sử dụng heuristic...")
        for i, speaker in enumerate(sorted(speakers)):
            genders[speaker] = "Male" if i % 2 == 0 else "Female"

    # ✅ Lưu kết quả
    print("   💾 Đang lưu kết quả...")
    save_start = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(genders, f, ensure_ascii=False, indent=2)
    save_time = time.time() - save_start
    print(f"   ✅ Lưu file: {save_time:.1f}s")

    elapsed = time.time() - start_time
    end_dt = datetime.now().strftime("%H:%M:%S")
    print(f"\n✅ Đã lưu kết quả giới tính vào {output_path}")
    print(f"⏱️  Thời gian gender classification: {elapsed:.2f}s")
    print(f"   ⏰ Kết thúc lúc: {end_dt}")

    # In tóm tắt
    print("\n   📊 Tóm tắt:")
    for speaker, gender in sorted(genders.items()):
        print(f"      {speaker}: {gender}")

    return output_path
