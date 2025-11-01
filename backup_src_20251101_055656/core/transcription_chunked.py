"""Transcribe file dài bằng cách chia nhỏ."""
import json
from pydub import AudioSegment
from transcribe import transcribe_audio
import os

def transcribe_long_audio(audio_path, output_path, chunk_length_minutes=10):
    """
    Chia file dài thành chunks nhỏ, transcribe riêng rồi ghép lại.
    
    Args:
        chunk_length_minutes: Độ dài mỗi chunk (phút)
    """
    print(f"🔪 Chia file thành chunks {chunk_length_minutes} phút...")
    
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = chunk_length_minutes * 60 * 1000
    
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"{audio_path}_chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    print(f"   ✅ Đã chia thành {len(chunks)} chunks")
    
    # Transcribe từng chunk
    all_segments = []
    time_offset = 0
    
    for idx, chunk_path in enumerate(chunks):
        print(f"\n📝 Xử lý chunk {idx+1}/{len(chunks)}...")
        
        chunk_output = chunk_path.replace(".wav", "_transcript.json")
        transcribe_audio(chunk_path, chunk_output)
        
        # Load kết quả
        with open(chunk_output, "r", encoding="utf-8") as f:
            chunk_result = json.load(f)
        
        # Điều chỉnh timestamps
        for seg in chunk_result.get("segments", []):
            seg["start"] += time_offset
            seg["end"] += time_offset
            all_segments.append(seg)
        
        time_offset += chunk_length_minutes * 60
        
        # Cleanup
        os.remove(chunk_path)
        os.remove(chunk_output)
    
    # Ghép kết quả
    final_result = {
        "text": " ".join(seg["text"] for seg in all_segments),
        "segments": all_segments,
        "language": "vi"
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Đã ghép {len(all_segments)} segments vào {output_path}")
    return output_path
