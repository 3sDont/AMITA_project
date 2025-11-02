# 🎙️ Audio Transcription & Speaker Diarization Pipeline

Pipeline tự động xử lý audio, nhận dạng người nói, phân loại giới tính và phân tích nội dung bằng AI.

## 📋 Tổng quan

Hệ thống bao gồm 5 giai đoạn xử lý tuần tự:

```
📁 Audio (MP3/WAV/M4A/...)
    ↓
🎵 Stage 0: Audio Preprocessing (16kHz mono, normalized, filtered)
    ↓
🎤 Stage 1: Whisper Transcription (faster-whisper)
    ↓
👥 Stage 2: Speaker Diarization (pyannote.audio 3.1)
    ↓
👤 Stage 2.5: Gender Classification (librosa pitch analysis)
    ↓
🔗 Stage 3: Combining Results (merge all data)
    ↓
🤖 Stage 4: LLM Analysis (Ollama - optional)
    ↓
📄 Final Output JSON + Text Reports
```

## 🚀 Cài đặt nhanh

### 1. Clone và cài đặt dependencies

```bash
# Clone hoặc download project
cd PyannoteTest

# Tạo virtual environment (khuyến nghị)
python -m venv .venv

# Kích hoạt environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Cài đặt packages
pip install -r requirements.txt
```

### 2. Cấu hình HuggingFace Token

Đăng ký token tại: https://huggingface.co/settings/tokens

Sửa file `config.py`:
```python
HF_TOKEN = "hf_your_token_here"
```

### 3. Cài đặt Ollama (Optional - cho LLM analysis)

Download tại: https://ollama.ai

```bash
# Pull model
ollama pull llama3.2
```

### 4. Chạy pipeline

```bash
# Sửa config.py để chọn file audio
python main.py
```

## ⚙️ Cấu hình

File `config.py` chứa tất cả cấu hình:

### Audio Settings
```python
AUDIO_FILE = r"D:\Data\your_audio.mp3"  # Đường dẫn file audio
OUTPUT_JSON = "transcription_output.json"  # File output
```

### GPU Settings
```python
USE_GPU = True  # Tự động dùng GPU nếu có, fallback về CPU
```

### Whisper Settings
```python
WHISPER_MODEL = "medium"  # tiny, base, small, medium, large
LANGUAGE = "vi"  # vi (Tiếng Việt), en (English), None (auto-detect)
```

### Diarization Settings
```python
MIN_SPEAKERS = None  # Số người tối thiểu (None = auto)
MAX_SPEAKERS = None  # Số người tối đa (None = auto)
```

### LLM Settings
```python
ENABLE_LLM_ANALYSIS = True  # Bật/tắt phân tích LLM
OLLAMA_MODEL = "llama3.2"  # Model Ollama
```

### Cache Paths
```python
WHISPER_CACHE = "outputs/whisper_output.json"
DIARIZATION_CACHE = "outputs/diarization_output.json"
GENDER_CACHE = "outputs/gender_output.json"
COMBINING_CACHE = "outputs/combining_output.json"
```

## 🏗️ Cấu trúc thư mục

```
PyannoteTest/
├── main.py                     # 🚀 File chính - chạy toàn bộ pipeline
├── config.py                   # ⚙️  Cấu hình tổng thể
├── utils.py                    # 🛠️ Hàm tiện ích (load/save JSON)
├── requirements.txt            # 📦 Dependencies
├── README.md                   # 📖 Tài liệu này
│
├── src/                        # 📁 Source modules
│   ├── audio_processor.py      # 🎵 Stage 0: Preprocessing
│   ├── whisper.py              # 🎤 Stage 1: Transcription
│   ├── diarization.py          # 👥 Stage 2: Speaker separation
│   ├── gender_classifier.py    # 👤 Stage 2.5: Gender detection
│   ├── combiner.py             # 🔗 Stage 3: Merge results
│   └── llm_applying.py         # 🤖 Stage 4: LLM analysis
│
├── outputs/                    # 📁 Cache và kết quả trung gian
│   ├── *_enhanced.wav          # Audio đã preprocessing
│   ├── whisper_output.json     # Kết quả transcription
│   ├── diarization_output.json # Kết quả diarization
│   ├── gender_output.json      # Kết quả gender classification
│   └── combining_output.json   # Kết quả merge tất cả
│
└── .venv/                      # Virtual environment
```

## 🎯 Cách sử dụng

### 1. Chạy toàn bộ pipeline (Khuyến nghị)

```bash
python main.py
```

Chạy tất cả 5 giai đoạn liên tiếp và tạo output cuối cùng.

### 2. Chạy từng giai đoạn riêng lẻ

```bash
# Stage 0: Audio preprocessing
python src/audio_processor.py

# Stage 1: Whisper transcription
python src/whisper.py

# Stage 2: Speaker diarization
python src/diarization.py

# Stage 2.5: Gender classification
python src/gender_classifier.py

# Stage 3: Combining results
python src/combiner.py

# Stage 4: LLM analysis (optional)
python src/llm_applying.py
```

### 3. Sử dụng trong code Python

```python
import config
from src import audio_processor
from src import whisper
from src import diarization

# Cấu hình
config.AUDIO_FILE = "my_audio.mp3"
config.WHISPER_MODEL = "large"

# Chạy từng bước
enhanced = audio_processor.enhance_audio(config.AUDIO_FILE)
whisper_result = whisper.transcribe_audio_optimized(enhanced)
diar_result = diarization.diarize_audio(enhanced)
```

## 📊 Output Format

### Final JSON Output
```json
{
  "metadata": {
    "audio_file": "D:\\Data\\meeting.mp3",
    "processed_at": "2025-11-02T14:30:00",
    "total_duration": 325.1,
    "number_of_speakers": 2,
    "number_of_segments": 44,
    "whisper_model": "medium",
    "language": "vi"
  },
  "speaker_statistics": {
    "SPEAKER_00": {
      "segments": 23,
      "duration": 180.5,
      "percentage": 55.5,
      "gender": "Female"
    },
    "SPEAKER_01": {
      "segments": 21,
      "duration": 144.6,
      "percentage": 44.5,
      "gender": "Female"
    }
  },
  "transcript": [
    {
      "start": 0.0,
      "end": 1.5,
      "speaker": "SPEAKER_00",
      "gender": "Female",
      "text": "Xin chào các bạn"
    }
  ]
}
```

### Cache Files (Intermediate)

**Audio Preprocessing Output:**
- File: `outputs/{basename}_enhanced.wav`
- Format: 16kHz mono WAV, normalized, high-pass filtered

**Whisper Output:** `outputs/whisper_output.json`
```json
[
  {
    "start_time": 0.0,
    "end_time": 1.5,
    "text": "Xin chào các bạn",
    "words": [...]
  }
]
```

**Diarization Output:** `outputs/diarization_output.json`
```json
[
  {
    "speaker": "SPEAKER_00",
    "start_time": 0.0,
    "end_time": 1.5
  }
]
```

**Gender Output:** `outputs/gender_output.json`
```json
{
  "SPEAKER_00": "Female",
  "SPEAKER_01": "Female"
}
```

**Combining Output:** `outputs/combining_output.json`
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 1.5,
    "text": "Xin chào các bạn",
    "gender": "Female"
  }
]
```

## 🔧 Chi tiết các modules

### 🎵 Stage 0: Audio Processor
**File:** `src/audio_processor.py`

**Chức năng:**
- Convert các format (MP3, M4A, AAC, OGG, FLAC) → WAV
- Resample về 16kHz mono (tối ưu cho speech models)
- Normalize volume (max 0.95)
- High-pass filter 80Hz (loại bỏ nhiễu tần số thấp)

**Tại sao cần:**
- Đảm bảo chất lượng đầu vào nhất quán cho tất cả models
- Giảm nhiễu, cải thiện độ chính xác
- Tránh lỗi format/sample rate không tương thích

**Output:** `outputs/{basename}_enhanced.wav`

### 🎤 Stage 1: Whisper Transcription
**File:** `src/whisper.py`

**Chức năng:**
- Transcribe audio thành text bằng faster-whisper
- Hỗ trợ tiếng Việt và nhiều ngôn ngữ khác
- GPU-accelerated (nếu có)
- Chunk processing để xử lý file lớn

**Models:** tiny, base, small, medium, large

**Output:** JSON với segments (start_time, end_time, text, words)

### 👥 Stage 2: Speaker Diarization
**File:** `src/diarization.py`

**Chức năng:**
- Phân đoạn audio theo người nói (Who spoke when?)
- Sử dụng pyannote.audio 3.1 (state-of-the-art)
- Tự động phát hiện số người nói
- GPU-accelerated

**Output:** JSON với segments (speaker, start_time, end_time)

### 👤 Stage 2.5: Gender Classification
**File:** `src/gender_classifier.py`

**Chức năng:**
- Phân loại giới tính dựa trên pitch (tần số cơ bản)
- Nam: 85-180 Hz, Nữ: 165-255 Hz
- Sử dụng librosa để trích xuất pitch features

**Output:** JSON mapping speaker → gender

### 🔗 Stage 3: Combiner
**File:** `src/combiner.py`

**Chức năng:**
- Merge kết quả từ Whisper + Diarization + Gender
- Map text vào đúng người nói
- Tạo timeline hoàn chỉnh với đầy đủ thông tin

**Output:** JSON với segments (speaker, start, end, text, gender)

### 🤖 Stage 4: LLM Analysis (Optional)
**File:** `src/llm_applying.py`

**Chức năng:**
- Phân tích nội dung cuộc hội thoại bằng Ollama
- Tạo summary, tổng hợp tasks/decisions
- Chạy riêng biệt, không ảnh hưởng pipeline chính

**Output:** 
- `outputs/dialog.txt` - Cuộc hội thoại đã format
- `outputs/meeting_summary.txt` - Tóm tắt
- `outputs/meeting_tasks.json` - Tasks/action items

## 💡 Tips & Best Practices

### Performance

1. **GPU:** Bật `USE_GPU = True` trong config.py
   - Whisper: 3-5x nhanh hơn
   - Diarization: 2-3x nhanh hơn

2. **Whisper Model:**
   - `tiny/base`: Nhanh nhưng kém chính xác
   - `small`: Cân bằng tốt cho tiếng Việt
   - `medium`: Chính xác cao (khuyến nghị)
   - `large`: Tốt nhất nhưng chậm

3. **Cache:** Các file cache cho phép chạy lại từng stage mà không cần xử lý từ đầu

### Accuracy

1. **Audio Quality:** File WAV chất lượng cao cho kết quả tốt nhất
2. **Speaker Count:** Set `MIN_SPEAKERS` và `MAX_SPEAKERS` nếu biết trước
3. **Language:** Chỉ định ngôn ngữ chính xác (`LANGUAGE = "vi"`)

### Troubleshooting

**Lỗi GPU:**
```
RuntimeError: CUDA out of memory
```
→ Set `USE_GPU = False` hoặc dùng model nhỏ hơn

**Lỗi HuggingFace Token:**
```
HTTPError: 401 Client Error: Unauthorized
```
→ Kiểm tra `HF_TOKEN` trong config.py

**Lỗi Ollama:**
```
Connection refused
```
→ Start Ollama service: `ollama serve`

**File MP3 không đọc được:**
→ Audio preprocessing sẽ tự động convert sang WAV

## 🔬 Thử nghiệm và Debug

### Test từng module riêng

```bash
# Test audio preprocessing
python src/audio_processor.py

# Test whisper với file enhanced
python src/whisper.py

# Test diarization
python src/diarization.py
```

### Xem cache files

```bash
# Windows
Get-Content outputs\whisper_output.json | ConvertFrom-Json | Format-List

# Linux/Mac
cat outputs/whisper_output.json | jq .
```

### So sánh models

```python
# Test nhiều Whisper models
for model in ["tiny", "base", "small", "medium"]:
    config.WHISPER_MODEL = model
    config.WHISPER_CACHE = f"outputs/whisper_{model}.json"
    whisper.transcribe_audio_optimized(audio_path)
```

## 📦 Dependencies

### Core
- **torch** (2.0+): Deep learning framework
- **torchaudio** (2.0+): Audio processing
- **pyannote.audio** (3.1+): Speaker diarization
- **faster-whisper** (0.8+): Fast transcription

### Audio Processing
- **librosa** (0.10+): Audio analysis
- **soundfile** (0.12+): Audio I/O
- **pydub** (0.25+): Audio conversion
- **scipy** (1.10+): Signal processing

### LLM (Optional)
- **ollama** (0.1+): Ollama Python client

### Utilities
- **numpy** (1.24+): Numerical computing
- **huggingface-hub** (0.15+): HuggingFace integration

## 🎓 Technical Details

### Audio Preprocessing
- **Sample Rate:** 16000 Hz (optimal for speech)
- **Channels:** Mono (single channel)
- **Format:** WAV PCM 16-bit
- **Filter:** Butterworth high-pass 80Hz, order 5
- **Normalization:** Peak normalization to 0.95

### Whisper Configuration
- **Beam Size:** 5 (default)
- **VAD Filter:** Enabled
- **Chunk Length:** 10 minutes (configurable)
- **Device:** CUDA if available, else CPU

### Diarization Configuration
- **Model:** pyannote/speaker-diarization-3.1
- **Segmentation:** Neural segmentation
- **Clustering:** Constrained clustering
- **Min Duration:** 0.5s (configurable)

### Gender Classification
- **Method:** Pitch-based analysis
- **Male Range:** 85-180 Hz
- **Female Range:** 165-255 Hz
- **Ambiguous Threshold:** 10% overlap
- **Feature:** Median fundamental frequency (F0)

## 📝 License & Credits

**Models Used:**
- Whisper: OpenAI (MIT License)
- Pyannote.audio: CNRS (MIT License)
- Ollama: Open source models

**Dependencies:**
See `requirements.txt` for full list

## 🤝 Contributing

Có thể mở rộng hoặc cải thiện:
1. Thêm models khác (Vosk, Wav2Vec2, etc.)
2. Hỗ trợ streaming processing
3. Web interface với FastAPI/Streamlit
4. Multi-language support
5. Real-time processing

## 📞 Support & Issues

**Lỗi thường gặp:**
- Cache files không tồn tại → Chạy các stage trước đó
- GPU out of memory → Giảm model size hoặc dùng CPU
- HuggingFace unauthorized → Kiểm tra token
- Ollama connection → Start Ollama service

**Performance:**
- File 10 phút: ~2-5 phút (GPU) hoặc 10-20 phút (CPU)
- Bottleneck: Whisper transcription (chiếm 60-70% thời gian)

---

**Version:** 2.0  
**Last Updated:** November 2, 2025  
**Python:** 3.8+  
**Platform:** Windows/Linux/Mac
