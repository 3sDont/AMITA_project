"""
Configuration file for AMITA Backend
All settings for audio processing pipeline
"""
from pathlib import Path

# ==================== PATHS ====================
BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

# ==================== WHISPER SETTINGS ====================
# 🎯 WHISPER_MODEL - faster-whisper official models
# Options: 
#   - "tiny" or "tiny.en" (very fast, ~1GB VRAM, 32x realtime)
#   - "base" or "base.en" (fast, ~1GB VRAM, 16x realtime)
#   - "small" or "small.en" (balanced, ~2GB VRAM, 6x realtime)
#   - "medium" or "medium.en" (good accuracy, ~5GB VRAM, 2x realtime) ✅ Recommended
#   - "large-v2" (high accuracy, ~10GB VRAM, 1x realtime)
#   - "large-v3" (best accuracy, ~10GB VRAM, 1x realtime)
# Note: Use ".en" suffix for English-only models (faster but English only)
WHISPER_MODEL = "large-v2"

LANGUAGE = "vi"  # ISO 639-1 code, e.g., "en" for English, "vi" for Vietnamese, or None for auto-detect
USE_GPU = False  # ✅ TEST: Disable GPU to check if CUDA is causing crash

# ⚙️ WHISPER_BEAM_SIZE - Balance giữa speed và accuracy
# - 1: Greedy decoding, rất nhanh, accuracy thấp (không khuyến nghị)
# - 3: Nhanh, accuracy tốt (cho file dài)
# - 5: Cân bằng (default, recommended) ✅
# - 10: Chậm, accuracy cao (cho file ngắn quan trọng)
WHISPER_BEAM_SIZE = 5

# ⚙️ WHISPER_VAD_FILTER - Voice Activity Detection
# True: Tự động bỏ qua đoạn im lặng (recommended cho file dài)
# False: Transcribe toàn bộ audio
WHISPER_VAD_FILTER = True

# ==================== DIARIZATION SETTINGS ====================
MIN_SPEAKERS = None  # None = auto-detect
MAX_SPEAKERS = None  # None = auto-detect

# ==================== HUGGING FACE SETTINGS ====================
# Required for PyAnnote diarization model
# Get token from: https://huggingface.co/settings/tokens
HF_TOKEN = None  # Will be loaded from .env file if not set here

# ==================== PROCESSING SETTINGS ====================
# ⚙️ CHUNK_DURATION_MINUTES - Quan trọng nhất cho file audio lớn
# - File ngắn (<30 phút): 10 phút
# - File trung bình (30-60 phút): 7 phút  
# - File dài (1-2 giờ): 5 phút
# - File rất dài (>2 giờ): 3 phút
# Trade-off: Nhỏ hơn = ít memory hơn nhưng lâu hơn
CHUNK_DURATION_MINUTES = 5

ENABLE_VAD = True  # Voice Activity Detection
ENABLE_GENDER = False  # Gender classification
ENABLE_SPELL_CHECK = False  # LLM spell checking and grammar correction
ENABLE_LLM_ANALYSIS = True  # LLM summary and tasks
DEBUG_MODE = True  # Save intermediate stage outputs for debugging

# ==================== PROCESSING MODES ====================
# 🎯 Three processing modes with different speed/accuracy trade-offs

PROCESSING_MODES = {
    "flash": {
        "name": "Flash Mode",
        "description": "⚡ Fastest - Quick results for short meetings",
        "whisper_model": "base",
        "whisper_beam_size": 3,
        "whisper_vad_filter": True,
        "chunk_duration_minutes": 10,
        "enable_vad": True,
        "enable_gender": False,
        "enable_spell_check": False,
        "enable_llm": True,
        "llm_detail_level": "brief",  # brief summary only
        "min_speakers": None,
        "max_speakers": None,
        "use_gpu": USE_GPU
    },
    "flow": {
        "name": "Flow Mode",
        "description": "⚖️ Balanced - Recommended for most meetings",
        "whisper_model": "small",
        "whisper_beam_size": 5,
        "whisper_vad_filter": True,
        "chunk_duration_minutes": 7,
        "enable_vad": True,
        "enable_gender": True,
        "enable_spell_check": True,
        "enable_llm": True,
        "llm_detail_level": "standard",  # full summary + tasks
        "min_speakers": None,
        "max_speakers": None,
        "use_gpu": USE_GPU
    },
    "deep": {
        "name": "Deep Mode",
        "description": "🎯 Most Accurate - Detailed analysis for important meetings",
        "whisper_model": "medium",
        "whisper_beam_size": 10,
        "whisper_vad_filter": True,
        "chunk_duration_minutes": 3,
        "enable_vad": True,
        "enable_gender": True,
        "enable_spell_check": True,
        "enable_llm": True,
        "llm_detail_level": "detailed",  # comprehensive analysis
        "min_speakers": MIN_SPEAKERS,
        "max_speakers": MAX_SPEAKERS,
        "use_gpu": USE_GPU
    }
}

# Default processing mode
DEFAULT_PROCESSING_MODE = "flow"

# ==================== PIPELINE CONFIG ====================
PIPELINE_CONFIG = {
    "chunk_duration_minutes": CHUNK_DURATION_MINUTES,
    "enable_vad": ENABLE_VAD,
    "enable_gender": ENABLE_GENDER,
    "enable_spell_check": ENABLE_SPELL_CHECK,
    "enable_llm": ENABLE_LLM_ANALYSIS,
    "min_speakers": MIN_SPEAKERS,
    "max_speakers": MAX_SPEAKERS,
    "debug_mode": DEBUG_MODE
}

# ==================== API SETTINGS ====================
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["http://localhost:5173"]  # Vite dev server

# ==================== FILE SETTINGS ====================
MAX_UPLOAD_SIZE_MB = 500
ALLOWED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm']

# ==================== OUTPUT SETTINGS ====================
MAX_TRANSCRIPT_SEGMENTS = 30  # Number of segments to send to UI
KEEP_OUTPUTS_DAYS = 7  # Auto-cleanup old outputs after N days

# ==================== AUDIO PROCESSING SETTINGS ====================
# Preprocessing thresholds
SAMPLE_RATE = 16000  # Target sample rate (Hz)
HIGH_PASS_FILTER_FREQ = 80  # Hz - Remove low frequency noise
NORMALIZE_TARGET = 0.95  # Target normalization level (0.0-1.0)
LARGE_FILE_THRESHOLD_MB = 500  # Use streaming mode for files larger than this

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
ENABLE_CONSOLE_LOGS = True
