"""File and path utilities."""
import os

def prepare_output_paths(audio_path: str):
    """Tạo thư mục output riêng cho từng file."""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # ✅ FIX: Loại bỏ suffix _temp, _enhanced nếu có
    base_name = base_name.replace('_temp', '').replace('_enhanced', '')
    
    output_dir = os.path.join("outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)

    suffix = base_name.split("_")[-1] if "_" in base_name else base_name
    return {
        "output_dir": output_dir,
        "diarization": os.path.join(output_dir, f"diarization_result_{suffix}.txt"),
        "transcription": os.path.join(output_dir, f"transcription_result_{suffix}.json"),
        "gender": os.path.join(output_dir, f"gender_result_{suffix}.json"),
        "combined": os.path.join(output_dir, f"combined_result_{suffix}.json")
    }
