"""Utility functions for file and path management."""
import os


def prepare_output_paths(audio_path: str):
    """
    Tạo thư mục output riêng cho từng file.
    
    Returns:
        dict: Đường dẫn đến các file output
    """
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Loại bỏ suffix không cần thiết
    base_name = base_name.replace('_temp', '').replace('_enhanced', '').replace('_audio', '')
    
    # Lưu vào data/output
    output_dir = os.path.join("data", "output", base_name)
    os.makedirs(output_dir, exist_ok=True)

    suffix = base_name.split("_")[-1] if "_" in base_name else base_name
    
    return {
        "output_dir": output_dir,
        "diarization": os.path.join(output_dir, f"diarization_result_{suffix}.txt"),
        "transcription": os.path.join(output_dir, f"transcription_result_{suffix}.json"),
        "gender": os.path.join(output_dir, f"gender_result_{suffix}.json"),
        "combined": os.path.join(output_dir, f"combined_result_{suffix}.json")
    }
