#!/usr/bin/env python3
# llm_applying.py
# Improved LLM meeting analysis pipeline (Ollama)
# Requirements: config.py (with ENABLE_LLM_ANALYSIS, COMBINING_CACHE, OLLAMA_MODEL), utils.py, ollama.chat

import sys
import os
import time
import json
import re
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Add project root to path so imports work when running script from /src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import config
from ollama import chat
import utils

# ✅ FIX: Import text preprocessor với error handling (TRƯỚC khi dùng logger)
PREPROCESSING_AVAILABLE = False
try:
    from text_preprocessor import TextPreprocessor
    PREPROCESSING_AVAILABLE = True
except ImportError:
    pass  # ✅ Không log warning ở đây vì logger chưa được init

# -------------------------
# Logging config
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_analysis")

# ✅ Bây giờ mới log warning nếu preprocessing không available
if not PREPROCESSING_AVAILABLE:
    logger.warning("⚠️  Text preprocessor not available. Using raw transcript.")

# -------------------------
# Transcript utilities
# -------------------------
def load_transcript(json_path: str, include_speaker=True, include_gender=True) -> str:
    """
    Load transcript từ JSON combining output (list hoặc {segments:[]})
    """
    data = utils.load_json(json_path)
    segments = data if isinstance(data, list) else data.get("segments") or data.get("transcript")

    if not segments:
        raise ValueError(f"Không tìm thấy segments trong file {json_path}")

    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "UNKNOWN")
        gender = seg.get("gender", "Unknown")

        if include_speaker and include_gender:
            emoji = "👨" if gender == "Male" else "👩" if gender == "Female" else "❓"
            lines.append(f"{emoji} {speaker} ({gender}): {text}")
        elif include_speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    return text.strip()


def save_dialog_file(json_path: str, output_path: str) -> List[str]:
    """
    Gộp lời thoại theo speaker → xuất file dialog.txt
    """
    data = utils.load_json(json_path)
    segments = data if isinstance(data, list) else data.get("segments") or data.get("transcript")

    dialog = []
    cur_spk, cur_gender, buffer = None, None, []

    for seg in segments:
        txt = seg.get("text", "").strip()
        if not txt:
            continue

        spk = seg.get("speaker", "UNKNOWN")
        gender = seg.get("gender", "Unknown")

        if spk == cur_spk:
            buffer.append(txt)
        else:
            if cur_spk:
                emoji = "👨" if cur_gender == "Male" else "👩" if cur_gender == "Female" else "❓"
                dialog.append(f"{emoji} {cur_spk} ({cur_gender}): {' '.join(buffer)}")

            cur_spk, cur_gender = spk, gender
            buffer = [txt]

    if cur_spk:
        emoji = "👨" if cur_gender == "Male" else "👩" if cur_gender == "Female" else "❓"
        dialog.append(f"{emoji} {cur_spk} ({cur_gender}): {' '.join(buffer)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== DIALOG ===\n\n")
        for line in dialog:
            f.write(line + "\n\n")
        f.write(f"Tổng lượt phát biểu: {len(dialog)}\n")

    logger.info(f"✅ Saved dialog → {output_path}")
    return dialog


# -------------------------
# LLM Utilities
# -------------------------
def safe_chat_call(model: str, messages: List[Dict[str, str]], retries=3, backoff=1.2):
    """
    Ollama call có retry
    """
    for attempt in range(1, retries + 1):
        try:
            return chat(model=model, messages=messages)
        except Exception as e:
            if attempt < retries:
                logger.warning(f"Ollama lỗi (lần {attempt}), retry...")
                time.sleep(backoff * attempt)
            else:
                raise e


def chunk_text(text: str, max_len=3500) -> List[str]:
    if len(text) <= max_len:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        part = text[start:end]

        nl = part.rfind("\n")
        if nl > 2000:
            end = start + nl

        chunks.append(text[start:end])
        start = end

    return chunks


def extract_json(response: str):
    """
    Extract JSON array from LLM output → tasks
    """
    # Tìm các block JSON dạng [] hoặc {}
    blocks = []

    def find_blocks(s, open_c, close_c):
        stack = []
        start = None
        out = []
        for i, c in enumerate(s):
            if c == open_c:
                if not stack:
                    start = i
                stack.append(c)
            elif c == close_c and stack:
                stack.pop()
                if not stack and start is not None:
                    out.append(s[start:i+1])
                    start = None
        return out

    blocks += find_blocks(response, "[", "]")
    blocks += find_blocks(response, "{", "}")

    for b in sorted(blocks, key=len, reverse=True):  # ưu tiên block dài nhất
        try:
            return json.loads(b)
        except:
            pass

    return None


# -------------------------
# Prompts
# -------------------------
def build_summary_prompt(chunk: str):
    return f"""
Bạn là trợ lý AI, nhiệm vụ của bạn là TÓM TẮT CUỘC HỌP theo ĐÚNG cấu trúc dưới đây.

📌 YÊU CẦU NGHIÊM NGẶT:
- Tóm tắt NGẮN GỌN (100–150 từ)
- CHỈ dựa trên nội dung thực tế trong cuộc họp (không bịa thêm)
- KHÔNG mô tả lại lời thoại của từng speaker
- KHÔNG dùng câu như “Speaker 01 nói rằng…”
- CHỈ dùng dạng **insight tổng hợp**
- BẮT BUỘC dùng đúng 3 phần dưới đây
- GIỮ NGUYÊN VĂN các cụm tên file như: combining_output_Recording

🎯 FORMAT OUTPUT (BẮT BUỘC):

1. Chủ đề chính của cuộc họp:
<viết 1 câu>

2. Các điểm quan trọng được thảo luận:
- <insight 1> (combining_output_Recording)
- <insight 2> (combining_output_Recording)
- <insight 3> (combining_output_Recording)
- <insight 4> (combining_output_Recording)

3. Quyết định/Kết luận:
<ghi “Không được đề cập” nếu không có>

---
HỘI THOẠI:
{chunk}
"""



def build_tasks_prompt(chunk: str):
    return f"""
Bạn là trợ lý AI. Nhiệm vụ của bạn là TRÍCH XUẤT DANH SÁCH CÔNG VIỆC (TASKS) từ cuộc họp.

📌 YÊU CẦU NGHIÊM NGẶT:
- Chỉ tạo task nếu cuộc họp thực sự đề cập hành động phải làm.
- KHÔNG được suy diễn, không bịa thêm người hoặc deadline.
- Không dùng dạng “Speaker 01 nói…”
- Chuyển nội dung hội thoại thành nhiệm vụ thực tế.
- Nếu không rõ người thực hiện → assigned_to = null
- Nếu không rõ thời hạn → deadline = null
- Output BẮT BUỘC là JSON ARRAY hợp lệ.

📌 FORMAT OUTPUT CHUẨN:

[
  {{
    "task": "Công việc cần thực hiện (ngắn gọn, đúng trọng tâm)",
    "assigned_to": "Tên người hoặc null",
    "deadline": "Thời gian cụ thể hoặc null"
  }}
]

📌 QUY TẮC XÁC ĐỊNH PRIORITY:
- high: task gấp hoặc có deadline rõ ràng
- medium: quan trọng nhưng không gấp
- low: mang tính hỗ trợ / không cần làm ngay

---
HỘI THOẠI:
{chunk}

JSON OUTPUT (CHỈ JSON, KHÔNG GIẢI THÍCH):
"""



# -------------------------
# Stage Functions
# -------------------------
def summarize(text: str, model: str, output_dir="outputs"):
    chunks = chunk_text(text)
    prompt = build_summary_prompt("\n---\n".join(chunks[:2]))

    with open(f"{output_dir}/prompt_summary.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    resp = safe_chat_call(model, [{"role": "user", "content": prompt}])
    content = resp["message"]["content"]
    return content.strip()


def extract_tasks_stage(text: str, model: str, output_dir="outputs"):
    chunks = chunk_text(text)
    prompt = build_tasks_prompt("\n---\n".join(chunks[:3]))

    with open(f"{output_dir}/prompt_tasks.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    resp = safe_chat_call(model, [{"role": "user", "content": prompt}])
    raw = resp["message"]["content"]

    parsed = extract_json(raw)
    if isinstance(parsed, list):
        return parsed
    return []


# -------------------------
# Main pipeline WITH PREPROCESSING
# -------------------------
def run_pipeline(json_path, model, output_dir="outputs", enable_preprocessing=True):
    """
    ✅ IMPROVED: Add 7-stage text preprocessing với fallback
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("▶ Load transcript + generate dialog")
    save_dialog_file(json_path, f"{output_dir}/dialog.txt")

    # ✅ Load RAW segments
    data = utils.load_json(json_path)
    segments = data if isinstance(data, list) else data.get("segments") or data.get("transcript")
    
    if not segments:
        raise ValueError(f"Không tìm thấy segments trong {json_path}")
    
    # ✅ RUN PREPROCESSING PIPELINE (với fallback)
    if enable_preprocessing and PREPROCESSING_AVAILABLE:
        try:
            logger.info("▶ Running 7-stage text preprocessing...")
            preprocessor = TextPreprocessor(
                enable_content_filter=True,
                max_length=5000
            )
            preprocessed = preprocessor.preprocess(segments)
            
            # Save stats
            with open(f"{output_dir}/preprocessing_stats.json", "w", encoding="utf-8") as f:
                json.dump(preprocessed["stats"], f, indent=2, ensure_ascii=False)
            
            summary_text = preprocessed["summary_input"]
            tasks_text = preprocessed["tasks_input"]
            
            logger.info(f"   ✅ Preprocessing done: {len(summary_text)} chars")
        except Exception as e:
            logger.warning(f"⚠️  Preprocessing failed: {e}. Using raw transcript.")
            enable_preprocessing = False
    
    # Fallback: Use raw transcript
    if not enable_preprocessing or not PREPROCESSING_AVAILABLE:
        logger.info("▶ Using RAW transcript")
        text = load_transcript(json_path)
        text = clean_text(text)
        summary_text = text
        tasks_text = text

    # Summary
    logger.info("▶ Tóm tắt cuộc họp")
    summary = summarize(summary_text, model, output_dir)
    with open(f"{output_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    # Tasks
    logger.info("▶ Trích xuất tasks")
    tasks = extract_tasks_stage(tasks_text, model, output_dir)
    with open(f"{output_dir}/tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info("✅ DONE")
    return {"summary": summary, "tasks": tasks}


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default=config.COMBINING_CACHE)
    p.add_argument("--model", "-m", default=config.OLLAMA_MODEL)
    p.add_argument("--out", "-o", default="outputs")
    p.add_argument("--no-preprocessing", action="store_true", help="Disable text preprocessing")  # ✅ ADD
    return p.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        result = run_pipeline(
            args.input, 
            args.model, 
            args.out,
            enable_preprocessing=not args.no_preprocessing  # ✅ ADD
        )

        print("\n===== SUMMARY PREVIEW =====\n")
        print(result["summary"])

        print("\n===== TASKS (JSON) =====\n")
        print(json.dumps(result["tasks"], ensure_ascii=False, indent=2))

    except Exception as e:
        logger.exception("LỖI:")
