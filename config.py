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
HF_TOKEN = "..."

# File audio đầu vào - Hỗ trợ: MP3, WAV, M4A, AAC, OGG, FLAC
# AUDIO_FILE = r"D:\Data\rgw-sxmw-fng-_2025-10-14-13_29-GMT-7_.mp3"
# AUDIO_FILE = r"D:\Data\amita_meetingrecord_2F.mp3"  # 2 nữ
#AUDIO_FILE = r"D:\Data\amita_meetingrecord_2F2M.mp3"  # 2 nam 2 nữ
#AUDIO_FILE = r"D:\Data\amita_meetingrecord_1F2M.mp3"  # 2 nam, 1 nữ
AUDIO_FILE = r"C:\Users\Admin\Downloads\Paper for seminar\data\input\Recording.m4a"  # Youtube
# AUDIO_FILE = r"C:\Users\Admin\Downloads\Paper for seminar\data\input\amita_meetingrecord_1F2M.mp3"

# File output cuối cùng (tạo bởi main.py)
OUTPUT_JSON = "transcription_Recording_output.json"

# ====================== GPU SETTINGS ======================
USE_GPU = True  # ✅ ENABLED - GPU RTX 3050 (4GB VRAM) available
# Kiểm tra GPU: python test_gpu.py
# ✅ PyTorch 2.7.1+cu118 with CUDA 11.8

# ====================== WHISPER SETTINGS ======================
# MODEL_TYPE: Loại model transcription
MODEL_TYPE = "whisper"  # "whisper" (OpenAI) - khuyến nghị

# Whisper Model Size - ✅ OPTIMIZED FOR 4GB VRAM
# tiny (39M)    - Nhanh nhất, kém chính xác
# base (74M)    - Cân bằng cho test
# small (244M)  - ✅ TỐT cho 4GB VRAM + tiếng Việt
# medium (769M) - ✅ KHUYẾN NGHỊ nếu không bị OOM
# large (1550M) - ❌ QUÁ LỚN cho 4GB VRAM
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
WHISPER_CACHE = "outputs\\whisper_output_Recording.json"              # Stage 1: Transcription
DIARIZATION_CACHE = "outputs\\diarization_output_Recording.json"      # Stage 2: Diarization
GENDER_CACHE = "outputs\\gender_output_Recording.json"                # Stage 2.5: Gender
COMBINING_CACHE = "outputs\\combining_output_Recording.json"          # Stage 3: Combined results
COMBINING_DETAILED_CACHE = "outputs\\combining_detailed_output_Recording.json"  # Detailed version

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
WHISPER_CHUNK_LENGTH = 10  # Minutes per chunk (auto-adjusted for long files)
WHISPER_BEAM_SIZE = 5      # Beam search size

# ✅ Memory & Performance Settings
ENABLE_CHECKPOINTING = True  # Resume từ checkpoint nếu bị gián đoạn
ENABLE_RETRY = True          # Retry khi chunk bị lỗi
MAX_RETRIES = 3              # Số lần retry tối đa
CHUNK_OVERLAP = 0.5          # Giây overlap giữa chunks (tránh mất text)

# ✅ Memory Management
AUTO_CLEANUP = True          # Tự động xóa chunk files
FORCE_GC = True              # Force garbage collection sau mỗi chunk

# ✅ Progress Tracking
SHOW_PROGRESS_BAR = True     # Hiển thị progress bar (cần rich/tqdm)
SHOW_ETA = True              # Hiển thị estimated time remaining

# ✅ Audio Quality Validation
VALIDATE_AUDIO_QUALITY = True  # Kiểm tra chất lượng audio trước xử lý
MIN_SNR_DB = 10               # Signal-to-Noise Ratio tối thiểu (dB)

# ✅ Parallel Processing (Experimental)
ENABLE_PARALLEL = False       # Xử lý song song nhiều chunks (cần nhiều GPU/CPU)
MAX_WORKERS = 2               # Số workers tối đa

# Diarization Advanced
DIAR_MIN_DURATION = 0.5    # Minimum segment duration in seconds

# ====================== NOTES ======================
# 1. Đảm bảo đã activate virtual environment: .venv\Scripts\Activate.ps1
# 2. HF_TOKEN phải valid và đã accept terms cho pyannote models
# 3. Ollama phải đang chạy nếu ENABLE_LLM_ANALYSIS = True
# 4. Kiểm tra GPU: python -c "import torch; print(torch.cuda.is_available())"
# 5. Chạy pipeline: python main.py
# 6. Chạy riêng từng module: python src/audio_processor.py, etc.


