"""
File cấu hình chung cho toàn bộ pipeline

📚 Xem hướng dẫn:
- README.md - Tài liệu đầy đủ
- README_MODULES.md - Chi tiết kỹ thuật từng module
- INSTALL.md - Hướng dẫn cài đặt

Pipeline gồm 5 giai đoạn:
0. Audio Preprocessing (audio_processor.py)
1. Whisper Transcription (whisper.py)
2. Speaker Diarization (diarization.py)
2.5. Gender Classification (gender_classifier.py)
3. Combining Results (combiner.py)
4. LLM Analysis - Optional (llm_applying.py)
"""

# ====================== AUDIO CONFIG ======================
# HuggingFace Token - Cần để sử dụng pyannote models
# Lấy token tại: https://huggingface.co/settings/tokens
# Cần accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN = "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"

# File audio đầu vào - Hỗ trợ: MP3, WAV, M4A, AAC, OGG, FLAC
# AUDIO_FILE = r"D:\Data\rgw-sxmw-fng-_2025-10-14-13_29-GMT-7_.mp3"
AUDIO_FILE = r"D:\Data\amita_meetingrecord_2F.mp3"  # 2 nữ
#AUDIO_FILE = r"D:\Data\amita_meetingrecord_2F2M.mp3"  # 2 nam 2 nữ
#AUDIO_FILE = r"D:\Data\amita_meetingrecord_1F2M.mp3"  # 2 nam, 1 nữ
#AUDIO_FILE = r"D:\Data\amita_ytb_1.mp3"  # Youtube

# File output cuối cùng (tạo bởi main.py)
OUTPUT_JSON = "transcription_2F_output.json"

# ====================== GPU SETTINGS ======================
USE_GPU = True  # True: Dùng GPU nếu có | False: Force dùng CPU
# Lưu ý: Nếu USE_GPU=True nhưng không có GPU, hệ thống tự động fallback về CPU
# Kiểm tra GPU: python -c "import torch; print(torch.cuda.is_available())"

# ====================== WHISPER SETTINGS ======================
# MODEL_TYPE: Loại model transcription
MODEL_TYPE = "whisper"  # "whisper" (OpenAI) - khuyến nghị

# Whisper Model Size - Càng lớn càng chính xác nhưng càng chậm
# tiny (39M)    - Nhanh nhất, kém chính xác
# base (74M)    - Cân bằng cho test
# small (244M)  - Tốt cho tiếng Việt
# medium (769M) - Chính xác cao (khuyến nghị)
# large (1550M) - Tốt nhất nhưng cần nhiều RAM/VRAM
WHISPER_MODEL = "medium"

# Ngôn ngữ - Nên chỉ định để tăng độ chính xác
LANGUAGE = "vi"  # vi (Tiếng Việt), en (English), None (auto-detect)

# ====================== PYANNOTE DIARIZATION SETTINGS ======================
# Pyannote/speaker-diarization-3.1 - State-of-the-art speaker diarization

# Số lượng người nói - Nên set nếu biết trước để tăng độ chính xác
MIN_SPEAKERS = None  # Số người tối thiểu (None = auto-detect)
MAX_SPEAKERS = None  # Số người tối đa (None = auto-detect)
# Ví dụ: MIN_SPEAKERS=2, MAX_SPEAKERS=4 → Tìm 2-4 người

# ====================== GENDER CLASSIFICATION SETTINGS ======================
# Gender classification dựa trên pitch analysis (librosa)
# Male: 85-180 Hz, Female: 165-255 Hz
# Không cần config thêm - tự động chạy trong pipeline

# ====================== LLM SETTINGS (OLLAMA ONLY) ======================
ENABLE_LLM_ANALYSIS = True  # Bật/tắt phân tích bằng LLM
# Set False nếu không cần LLM hoặc chưa cài Ollama

# Ollama Model - Phải pull trước: ollama pull llama3.2
OLLAMA_MODEL = "llama3.2"  # Khác: mistral, codellama, gemma, phi, qwen, etc.
# Xem models có sẵn: ollama list

# ====================== CACHE PATHS ======================
# Các file cache trung gian - Tự động tạo trong thư mục outputs/
# Stage 0: Audio preprocessing → outputs/{basename}_enhanced.wav (auto-generated)
WHISPER_CACHE = "outputs\\whisper_output_2F.json"              # Stage 1: Transcription
DIARIZATION_CACHE = "outputs\\diarization_output_2F.json"      # Stage 2: Diarization
GENDER_CACHE = "outputs\\gender_output_2F.json"                # Stage 2.5: Gender
COMBINING_CACHE = "outputs\\combining_output_2F.json"          # Stage 3: Combined results
COMBINING_DETAILED_CACHE = "outputs\\combining_detailed_output_2F.json"  # Detailed version

# Stage 4: LLM outputs (tạo bởi llm_applying.py)
# - outputs/dialog.txt
# - outputs/meeting_summary.txt
# - outputs/meeting_tasks.json

# ====================== ADVANCED SETTINGS ======================
# Không cần thay đổi trừ khi cần tùy chỉnh chi tiết

# Audio Preprocessing
AUDIO_SAMPLE_RATE = 16000  # Hz - Optimal for speech
AUDIO_CHANNELS = 1         # Mono
HIGH_PASS_CUTOFF = 80      # Hz - Remove low-frequency noise

# Whisper Advanced
WHISPER_CHUNK_LENGTH = 10  # Minutes per chunk
WHISPER_BEAM_SIZE = 5      # Beam search size

# Diarization Advanced
DIAR_MIN_DURATION = 0.5    # Minimum segment duration in seconds

# ====================== NOTES ======================
# 1. Đảm bảo đã activate virtual environment: .venv\Scripts\Activate.ps1
# 2. HF_TOKEN phải valid và đã accept terms cho pyannote models
# 3. Ollama phải đang chạy nếu ENABLE_LLM_ANALYSIS = True
# 4. Kiểm tra GPU: python -c "import torch; print(torch.cuda.is_available())"
# 5. Chạy pipeline: python main.py
# 6. Chạy riêng từng module: python src/audio_processor.py, etc.


