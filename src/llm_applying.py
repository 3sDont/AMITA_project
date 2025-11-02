"""
GIAI ĐOẠN 4: Phân tích bằng LLM (Ollama only)
- Input: Segments với text đầy đủ từ combining results
- Output: Analysis (summary, tasks, decisions, etc.)
"""
import sys
import os
# Thêm thư mục gốc vào sys.path để import config và utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')  # Tắt warnings
import json
import re
from typing import List, Dict, Any
import config
from ollama import chat
import utils


# === STEP 1: Đọc file JSON combining output ===
def load_transcript(json_path, include_speaker=True, include_gender=True):
    """
    Load transcript từ combining output.
    Format mới: list of segments với keys: speaker, start, end, text, gender
    """
    data = utils.load_json(json_path)

    # Xử lý cả 2 trường hợp: list trực tiếp hoặc dict với key "transcript"/"segments"
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict):
        segments = data.get("segments", data.get("transcript", []))
    else:
        raise ValueError(f"Format không hợp lệ trong file {json_path}")

    if not segments:
        raise ValueError(f"Không tìm thấy segments trong file {json_path}")

    # Gộp các đoạn text có nội dung
    text_segments = []
    for seg in segments:
        if not seg.get("text", "").strip():
            continue
        
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        
        if include_speaker and include_gender:
            # Hiển thị cả speaker và gender
            gender = seg.get("gender", "Unknown")
            gender_emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
            text_segments.append(f"{gender_emoji} {speaker} ({gender}): {text}")
        elif include_speaker:
            # Chỉ hiển thị speaker
            text_segments.append(f"{speaker}: {text}")
        else:
            # Chỉ lấy text
            text_segments.append(text)
    
    combined_text = "\n".join(text_segments) if include_speaker else " ".join(text_segments)
    return combined_text


def clean_text(text: str) -> str:
    """Làm sạch text trước khi phân tích"""
    text = re.sub(r'\s+', ' ', text)  # Xóa whitespace dư thừa
    text = re.sub(r'(.)\1{3,}', r'\1', text)  # Xóa ký tự lặp >3 lần
    return text.strip()


def save_dialog_file(json_path: str, output_path: str = "outputs/dialog.txt"):
    """
    Lưu dialog ra file riêng với format đẹp
    
    Args:
        json_path: Đường dẫn file JSON chứa transcript
        output_path: Đường dẫn file output (mặc định: outputs/dialog.txt)
    """
    data = utils.load_json(json_path)

    # Xử lý cả 2 trường hợp: list trực tiếp hoặc dict với key "segments"/"transcript"
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict):
        segments = data.get("segments", data.get("transcript", []))
    else:
        raise ValueError(f"Format không hợp lệ trong file {json_path}")

    if not segments:
        raise ValueError(f"Không tìm thấy segments trong file {json_path}")

    # Gộp các câu nói liên tiếp của cùng 1 speaker
    dialog_lines = []
    current_speaker = None
    current_gender = None
    current_text = []
    
    for seg in segments:
        if not seg.get("text", "").strip():
            continue
            
        speaker = seg.get("speaker", "UNKNOWN")
        gender = seg.get("gender", "Unknown")
        text = seg["text"].strip()
        
        if speaker == current_speaker:
            # Cùng speaker, gộp text
            current_text.append(text)
        else:
            # Speaker mới, lưu câu nói cũ
            if current_speaker is not None:
                combined = " ".join(current_text)
                gender_emoji = "👨" if current_gender == "Male" else "👩" if current_gender == "Female" else "❓"
                dialog_lines.append(f"{gender_emoji} {current_speaker} ({current_gender}): {combined}\n")
            
            # Bắt đầu câu nói mới
            current_speaker = speaker
            current_gender = gender
            current_text = [text]
    
    # Lưu câu nói cuối cùng
    if current_speaker is not None and current_text:
        combined = " ".join(current_text)
        gender_emoji = "👨" if current_gender == "Male" else "👩" if current_gender == "Female" else "❓"
        dialog_lines.append(f"{gender_emoji} {current_speaker} ({current_gender}): {combined}\n")
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ghi ra file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DIALOG - HỘI THOẠI CUỘC HỌP\n")
        f.write("=" * 70 + "\n\n")
        
        for line in dialog_lines:
            f.write(line)
            f.write("\n")  # Thêm dòng trống giữa các câu nói
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Tổng số lượt phát biểu: {len(dialog_lines)}\n")
        f.write("=" * 70 + "\n")
    
    print(f"💾 Đã lưu dialog vào: {output_path}")
    return dialog_lines

# === STEP 3: Gọi Ollama để tóm tắt ===
def summarize_meeting(text, model="llama3.2", save_prompt=True, output_dir="outputs"):
    prompt = f"""
Bạn là thư ký ghi biên bản cuộc họp.
Hãy đọc đoạn hội thoại dưới đây và viết **tóm tắt ngắn gọn** bằng tiếng Việt, bao gồm:
1. Chủ đề cuộc họp
2. Các ý chính được thảo luận
3. Các quyết định / kết luận chính
4. Các bước tiếp theo (nếu có)

---
{text}
"""

    # ✅ In prompt ra màn hình (bỏ phần dialog, chỉ hiển thị instruction)
    print("\n==========================")
    print("📥 PROMPT GỬI ĐẾN LLM (SUMMARY):")
    print("==========================")
    instruction = prompt.split("---")[0]
    print(instruction)
    print("--- [Dialog đã được tách ra file dialog.txt] ---")
    print("==========================\n")

    # ✅ Lưu prompt ra file (không bao gồm dialog)
    if save_prompt:
        prompt_path = os.path.join(output_dir, "llm_input_prompt_summary.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("PROMPT TÓM TẮT CUỘC HỌP\n")
            f.write("=" * 70 + "\n\n")
            f.write(instruction)
            f.write("\n\n--- Xem nội dung dialog trong file dialog.txt ---\n")
        print(f"💾 Đã lưu prompt tóm tắt vào: {prompt_path}")
        print("   (Dialog được lưu riêng trong file dialog.txt)\n")

    response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


# === STEP 4: Gọi Ollama để trích xuất task list ===
def extract_tasks(text, model="llama3.2", save_prompt=True, output_dir="outputs"):
    prompt = f"""
Đọc đoạn hội thoại dưới đây và trích xuất các **công việc cần làm (tasks)**.
Trả về dưới dạng JSON list, mỗi task gồm:
- task: mô tả công việc
- assigned_to: người phụ trách (nếu có)
- deadline: thời hạn (nếu có)
- status: luôn là "pending"

---
{text}
"""

    print("\n==========================")
    print("📥 PROMPT GỬI ĐẾN LLM (TASK EXTRACTION):")
    print("==========================")
    instruction = prompt.split("---")[0]
    print(instruction)
    print("--- [Dialog đã được tách ra file dialog.txt] ---")
    print("==========================\n")

    if save_prompt:
        prompt_path = os.path.join(output_dir, "llm_input_prompt_tasks.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("PROMPT TRÍCH XUẤT TASKS\n")
            f.write("=" * 70 + "\n\n")
            f.write(instruction)
            f.write("\n\n--- Xem nội dung dialog trong file dialog.txt ---\n")
        print(f"💾 Đã lưu prompt trích xuất task vào: {prompt_path}")
        print("   (Dialog được lưu riêng trong file dialog.txt)\n")

    response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


# === MAIN ===
if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🧠 GIAI ĐOẠN 4: LLM ANALYSIS")
        print("="*80 + "\n")
        
        # Kiểm tra config
        if not config.ENABLE_LLM_ANALYSIS:
            print("⚠️  LLM Analysis đã TẮT trong config.py")
            print("💡 Để bật: đặt ENABLE_LLM_ANALYSIS = True")
            sys.exit(0)
        
        # Kiểm tra file combining có tồn tại không
        json_path = config.COMBINING_CACHE
        if not os.path.exists(json_path):
            print(f"❌ Lỗi: Không tìm thấy file combining: {json_path}")
            print(f"� Vui lòng chạy src/combiner.py trước!")
            sys.exit(1)
        
        # Tạo thư mục outputs nếu chưa có
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Input:  {json_path}")
        print(f"⚙️  Config: Ollama Model = {config.OLLAMA_MODEL}\n")
        
        print("="*80)
        
        # Lưu dialog riêng trước
        print("\n💬 Đang tạo file dialog...")
        dialog_path = os.path.join(output_dir, "dialog.txt")
        save_dialog_file(json_path, dialog_path)
        
        # Load transcript để gửi cho LLM
        print("\n📖 Đang load transcript...")
        text = load_transcript(json_path, include_speaker=True, include_gender=True)
        text = clean_text(text)
        print(f"   ✅ Đã load {len(text)} ký tự")

        print("\n" + "="*80)
        print("🧠 Đang tóm tắt nội dung cuộc họp bằng Ollama...")
        print("="*80)
        summary = summarize_meeting(text, model=config.OLLAMA_MODEL, output_dir=output_dir)
        
        print("\n" + "="*80)
        print("📋 MEETING SUMMARY")
        print("="*80)
        print(summary)

        print("\n" + "="*80)
        print("📋 Đang trích xuất task list...")
        print("="*80)
        tasks = extract_tasks(text, model=config.OLLAMA_MODEL, output_dir=output_dir)
        
        print("\n" + "="*80)
        print("✅ TASKS")
        print("="*80)
        print(tasks)

        # Lưu ra file kết quả
        print("\n💾 Đang lưu kết quả...")
        summary_path = os.path.join(output_dir, "meeting_summary.txt")
        tasks_path = os.path.join(output_dir, "meeting_tasks.json")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        with open(tasks_path, "w", encoding="utf-8") as f:
            f.write(tasks)

        print("\n" + "="*80)
        print("✅ HOÀN THÀNH GIAI ĐOẠN 4")
        print("="*80)
        print("\n💾 Đã lưu các file vào thư mục outputs/:")
        print(f"   📄 dialog.txt                    - Hội thoại đầy đủ")
        print(f"   📄 meeting_summary.txt           - Tóm tắt cuộc họp")
        print(f"   📄 meeting_tasks.json            - Danh sách công việc")
        print(f"   📄 llm_input_prompt_summary.txt  - Prompt tóm tắt")
        print(f"   📄 llm_input_prompt_tasks.txt    - Prompt trích xuất task")
        
        print("\n" + "="*80)
        print("🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print("="*80)
        print("\n📊 Tổng quan các giai đoạn:")
        print("   ✅ Giai đoạn 1: Whisper Transcription")
        print("   ✅ Giai đoạn 2: Speaker Diarization")
        print("   ✅ Giai đoạn 2.5: Gender Classification")
        print("   ✅ Giai đoạn 3: Combining Results")
        print("   ✅ Giai đoạn 4: LLM Analysis")
        print("\n💡 Các kết quả đã lưu trong thư mục outputs/")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
