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
WHISPER_MODEL = "NhutP/ViWhisper-medium"
LANGUAGE = "vi"
USE_GPU = True  # Set to False to force CPU

# ==================== DIARIZATION SETTINGS ====================
MIN_SPEAKERS = None  # None = auto-detect
MAX_SPEAKERS = None  # None = auto-detect

# ==================== PROCESSING SETTINGS ====================
CHUNK_DURATION_MINUTES = 10  # Duration of each audio chunk
ENABLE_VAD = True  # Voice Activity Detection
ENABLE_GENDER = True  # Gender classification
ENABLE_LLM_ANALYSIS = True  # LLM summary and tasks

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

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
ENABLE_CONSOLE_LOGS = True
