"""
Cache management for faster processing.
"""
import json
import os
import hashlib


def get_file_hash(filepath):
    """Tính MD5 hash của file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cache_path(audio_path, step_name):
    """Tạo path cho cache file."""
    file_hash = get_file_hash(audio_path)
    cache_dir = os.path.join("cache", file_hash[:8])
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{step_name}.json")


def load_cache(audio_path, step_name):
    """Load kết quả từ cache (nếu có)."""
    cache_path = get_cache_path(audio_path, step_name)
    if os.path.exists(cache_path):
        print(f"   💾 Tìm thấy cache cho {step_name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(audio_path, step_name, data):
    """Lưu kết quả vào cache."""
    cache_path = get_cache_path(audio_path, step_name)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   💾 Đã cache {step_name} tại {cache_path}")


def clear_cache(audio_path=None):
    """Xóa cache (tất cả hoặc của 1 file)."""
    if audio_path:
        file_hash = get_file_hash(audio_path)
        cache_dir = os.path.join("cache", file_hash[:8])
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"   🗑️  Đã xóa cache cho {audio_path}")
    else:
        import shutil
        if os.path.exists("cache"):
            shutil.rmtree("cache")
            print("   🗑️  Đã xóa toàn bộ cache")
