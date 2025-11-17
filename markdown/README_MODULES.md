# 📚 Hướng dẫn Module Pipeline (Chi tiết kỹ thuật)

> 📖 **Để xem hướng dẫn đầy đủ, xem [README.md](README.md)**

Dự án đã được tách thành các module riêng biệt trong thư mục `src/`, mỗi module xử lý một giai đoạn cụ thể.

## 🏗️ Cấu trúc dự án mới

```
PyannoteTest/
├── main.py                        # 🚀 File chính chạy toàn bộ pipeline
├── config.py                      # ⚙️  Cấu hình chung
├── utils.py                       # 🛠️ Hàm tiện ích (load/save JSON)
├── requirements.txt               # 📦 Dependencies
├── README.md                      # 📖 Tài liệu chính
├── README_MODULES.md              # 📖 File này (chi tiết kỹ thuật)
│
├── src/                           # 📁 Source modules
│   ├── audio_processor.py         # � Stage 0: Audio preprocessing
│   ├── whisper.py                 # 🎤 Stage 1: Whisper transcription
│   ├── diarization.py             # � Stage 2: Speaker diarization
│   ├── gender_classifier.py       # � Stage 2.5: Gender classification
│   ├── combiner.py                # 🔗 Stage 3: Combining results
│   └── llm_applying.py            # 🤖 Stage 4: LLM analysis (optional)
│
└── outputs/                       # 📁 Cache và output files
    ├── *_enhanced.wav             # Audio đã preprocessing
    ├── whisper_output.json        # Cache Whisper
    ├── diarization_output.json    # Cache Diarization
    ├── gender_output.json         # Cache Gender
    ├── combining_output.json      # Cache Combining
    ├── dialog.txt                 # LLM output
    ├── meeting_summary.txt        # LLM output
    └── meeting_tasks.json         # LLM output
```

## 🎯 Cách sử dụng

### 1️⃣ Chạy toàn bộ pipeline (Khuyến nghị)

```bash
python main.py
```

Chạy tất cả 5 giai đoạn liên tiếp:
0. Audio preprocessing (convert, normalize, filter)
1. Transcribe audio với Whisper
2. Phân đoạn người nói với Pyannote
2.5. Phân loại giới tính (Gender classification)
3. Merge tất cả kết quả
4. Phân tích bằng LLM (nếu được bật)

### 2️⃣ Chạy từng giai đoạn riêng lẻ

**Giai đoạn 0: Audio Preprocessing**
```bash
python src/audio_processor.py
```
- Input: File audio bất kỳ (MP3, M4A, WAV, etc.)
- Output: `outputs/{basename}_enhanced.wav` (16kHz mono, normalized, filtered)
- Tại sao: Đảm bảo chất lượng đầu vào nhất quán cho tất cả models

**Giai đoạn 1: Transcribe**
```bash
python src/whisper.py
```
- Input: Audio đã preprocessing (hoặc file gốc)
- Output: `outputs/whisper_output.json`
- Chứa: segments với start_time, end_time, text, words

**Giai đoạn 2: Diarization**
```bash
python src/diarization.py
```
- Input: Audio đã preprocessing (hoặc file gốc)
- Output: `outputs/diarization_output.json`
- Chứa: segments với speaker, start_time, end_time

**Giai đoạn 2.5: Gender Classification**
```bash
python src/gender_classifier.py
```
- Input: Audio đã preprocessing + Diarization output
- Output: `outputs/gender_output.json`
- Chứa: mapping từ speaker → gender (Male/Female/Unknown)
- Method: Pitch analysis (librosa)

**Giai đoạn 3: Combining**
```bash
python src/combiner.py
```
- Input: Whisper + Diarization + Gender outputs
- Output: `outputs/combining_output.json`
- Chứa: segments với speaker, start, end, text, gender

**Giai đoạn 4: LLM Analysis**
```bash
python src/llm_applying.py
```
- Input: Combining output
- Output: 
  - `outputs/dialog.txt` - Cuộc hội thoại đã format
  - `outputs/meeting_summary.txt` - Tóm tắt
  - `outputs/meeting_tasks.json` - Tasks và action items
- Lưu ý: Cần Ollama đang chạy

## ⚙️ Cấu hình

Chỉnh sửa file `config.py`:

```python
# File audio
AUDIO_FILE = r"D:\Data\your_audio.mp3"
OUTPUT_JSON = "output.json"

# GPU
USE_GPU = True  # Auto fallback to CPU if no GPU

# Whisper settings
WHISPER_MODEL = "medium"  # tiny, base, small, medium, large
LANGUAGE = "vi"  # vi, en, hoặc None (auto)

# Diarization
MIN_SPEAKERS = None  # Số người tối thiểu (None = auto)
MAX_SPEAKERS = None  # Số người tối đa (None = auto)

# LLM settings
ENABLE_LLM_ANALYSIS = True  # Bật/tắt phân tích LLM
OLLAMA_MODEL = "llama3.2"  # Model Ollama

# Cache paths (tự động tạo trong outputs/)
WHISPER_CACHE = "outputs/whisper_output.json"
DIARIZATION_CACHE = "outputs/diarization_output.json"
GENDER_CACHE = "outputs/gender_output.json"
COMBINING_CACHE = "outputs/combining_output.json"
```

## 📁 File cache

Các file cache được tạo ra trong thư mục `outputs/` để có thể chạy lại từng bước mà không cần xử lý lại từ đầu:

| File | Giai đoạn | Format | Chứa gì |
|------|-----------|--------|---------|
| `*_enhanced.wav` | Stage 0 | WAV | Audio 16kHz mono, normalized, filtered |
| `whisper_output.json` | Stage 1 | JSON | Transcription segments |
| `diarization_output.json` | Stage 2 | JSON | Speaker segments |
| `gender_output.json` | Stage 2.5 | JSON | Speaker → Gender mapping |
| `combining_output.json` | Stage 3 | JSON | Full transcript với gender |
| `dialog.txt` | Stage 4 | Text | Formatted conversation |
| `meeting_summary.txt` | Stage 4 | Text | AI summary |
| `meeting_tasks.json` | Stage 4 | JSON | Tasks/action items |

## 🔄 Workflow linh hoạt

### Chạy lại một giai đoạn cụ thể

Nếu muốn thử nghiệm với cấu hình khác nhau:

```bash
# Thử model Whisper khác mà không cần chạy lại diarization
# 1. Sửa WHISPER_MODEL trong config.py
# 2. Chạy:
python src/whisper.py

# Sau đó chạy lại combining
python src/combiner.py
```

### Thử nghiệm LLM khác nhau

```bash
# Chạy các bước 0-3 một lần
python src/audio_processor.py
python src/whisper.py
python src/diarization.py
python src/gender_classifier.py
python src/combiner.py

# Thử nhiều LLM models mà không cần xử lý lại audio
# Sửa OLLAMA_MODEL trong config.py, sau đó:
python src/llm_applying.py
```

### Test audio preprocessing

```bash
# Test với các audio formats khác nhau
python src/audio_processor.py
# → Tự động convert MP3/M4A/AAC/OGG → WAV 16kHz mono
```

## 🛠️ Import module vào code khác

Các module có thể được import và sử dụng trong code khác:

```python
import config
from src import audio_processor
from src import whisper
from src import diarization
from src import gender_classifier
from src import combiner

# Tùy chỉnh config
config.WHISPER_MODEL = "large"
config.AUDIO_FILE = "my_audio.mp3"

# Chạy từng bước
enhanced = audio_processor.enhance_audio(config.AUDIO_FILE)
whisper_result = whisper.transcribe_audio_optimized(audio_path=enhanced)
diar_result = diarization.diarize_audio(audio_path=enhanced)
gender_result = gender_classifier.classify_gender(audio_path=enhanced)
combined = combiner.combine_results()

# Xử lý kết quả
for segment in combined:
    print(f"{segment['speaker']} ({segment['gender']}): {segment['text']}")
```

## 📊 Output format

### Stage 0: Audio Preprocessing Output
```
outputs/my_audio_enhanced.wav
```
- Format: WAV PCM 16-bit
- Sample Rate: 16000 Hz
- Channels: 1 (mono)
- Normalized: Peak at 0.95
- Filtered: High-pass 80Hz

### Stage 1: Whisper Output (`whisper_output.json`)
```json
[
  {
    "start_time": 0.0,
    "end_time": 1.5,
    "text": "Xin chào các bạn",
    "words": [
      {"word": "Xin", "start": 0.0, "end": 0.3},
      {"word": "chào", "start": 0.3, "end": 0.6}
    ]
  }
]
```

### Stage 2: Diarization Output (`diarization_output.json`)
```json
[
  {
    "speaker": "SPEAKER_00",
    "start_time": 0.0,
    "end_time": 1.5
  },
  {
    "speaker": "SPEAKER_01",
    "start_time": 1.5,
    "end_time": 3.0
  }
]
```

### Stage 2.5: Gender Output (`gender_output.json`)
```json
{
  "SPEAKER_00": "Female",
  "SPEAKER_01": "Female"
}
```

### Stage 3: Combining Output (`combining_output.json`)
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 1.5,
    "text": "Xin chào các bạn",
    "gender": "Female"
  },
  {
    "speaker": "SPEAKER_01",
    "start": 1.5,
    "end": 3.0,
    "text": "Chào em",
    "gender": "Female"
  }
]
```

### Stage 4: LLM Outputs

**dialog.txt:**
```
👩 SPEAKER_00 (Female): Xin chào các bạn
👩 SPEAKER_01 (Female): Chào em
```

**meeting_summary.txt:**
```
Cuộc họp bàn về...
- Key point 1
- Key point 2
```

**meeting_tasks.json:**
```json
{
  "tasks": [
    {
      "task": "Hoàn thành báo cáo",
      "assignee": "SPEAKER_00",
      "deadline": "2025-11-10"
    }
  ]
}
```

### Final Output (main.py tạo)
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
```

## 💡 Tips

1. **Chạy nhanh hơn**: Tắt LLM analysis nếu chỉ cần transcript
   ```python
   ENABLE_LLM_ANALYSIS = False
   ```

2. **Xử lý nhiều file**: Viết loop để xử lý nhiều audio
   ```python
   audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]
   for audio in audio_files:
       config.AUDIO_FILE = audio
       basename = os.path.splitext(os.path.basename(audio))[0]
       config.OUTPUT_JSON = f"outputs/{basename}_output.json"
       # Chạy pipeline...
   ```

3. **Debug**: Chạy từng step để kiểm tra kết quả trung gian
   ```bash
   python src/audio_processor.py
   # Kiểm tra outputs/*_enhanced.wav
   python src/whisper.py
   # Kiểm tra outputs/whisper_output.json
   ```

4. **Tiết kiệm thời gian**: 
   - Cache files cho phép chạy lại từng stage mà không cần xử lý từ đầu
   - Audio preprocessing chạy 1 lần, tất cả stages dùng chung enhanced audio

5. **GPU Performance**:
   - Whisper medium + GPU: ~3-5x nhanh hơn CPU
   - Diarization + GPU: ~2-3x nhanh hơn CPU
   - File 10 phút: ~2-5 phút (GPU) vs 10-20 phút (CPU)

## 🐛 Troubleshooting

**Lỗi "File not found" khi chạy stage 3/4:**
```
FileNotFoundError: outputs/whisper_output.json
```
→ Chạy các bước trước đó trước (stage 0, 1, 2, 2.5)

**Lỗi khi load LLM model:**
```
Connection refused: Ollama
```
→ Start Ollama service: `ollama serve`
→ Hoặc tắt LLM: `ENABLE_LLM_ANALYSIS = False`

**File MP3 không đọc được:**
```
RuntimeError: Error loading audio
```
→ Audio preprocessing sẽ tự động convert sang WAV
→ Hoặc cài ffmpeg: https://ffmpeg.org/download.html

**GPU out of memory:**
```
RuntimeError: CUDA out of memory
```
→ Set `USE_GPU = False` hoặc dùng model nhỏ hơn (`WHISPER_MODEL = "small"`)

**Import error:**
```
ModuleNotFoundError: No module named 'config'
```
→ Đảm bảo chạy từ thư mục gốc PyannoteTest/
→ Hoặc activate virtual environment: `.venv\Scripts\Activate.ps1`

**Diarization không chính xác:**
→ Set MIN_SPEAKERS và MAX_SPEAKERS nếu biết số người
→ Sử dụng audio chất lượng cao hơn
→ Chạy audio preprocessing trước (stage 0)

**Gender classification sai:**
→ Method dựa trên pitch, có thể sai với giọng đặc biệt
→ Threshold: Male 85-180Hz, Female 165-255Hz
→ Vùng overlap (165-180Hz) có thể gây nhầm lẫn

## 📞 Chi tiết kỹ thuật

### Audio Preprocessing (Stage 0)
- **Sample Rate:** 16000 Hz (optimal for speech models)
- **Channels:** Mono (1 channel)
- **Bit Depth:** 16-bit PCM
- **Filter Type:** Butterworth high-pass, order 5, cutoff 80Hz
- **Normalization:** Peak normalization to 0.95 max amplitude
- **Supported Inputs:** MP3, M4A, AAC, OGG, FLAC, WAV

### Whisper (Stage 1)
- **Engine:** faster-whisper (CTranslate2)
- **Models:** tiny (39M), base (74M), small (244M), medium (769M), large (1550M)
- **Beam Size:** 5 (configurable)
- **VAD:** Enabled by default
- **Device:** CUDA if available, else CPU
- **Chunk Processing:** 10 minutes per chunk (configurable)

### Diarization (Stage 2)
- **Model:** pyannote/speaker-diarization-3.1
- **Method:** Neural segmentation + constrained clustering
- **Min Segment Duration:** 0.5s (configurable)
- **Max Speakers:** Auto-detect or manual specification
- **Device:** CUDA if available, else CPU

### Gender Classification (Stage 2.5)
- **Method:** Pitch-based analysis using librosa
- **Feature:** Fundamental frequency (F0) extraction
- **Male Range:** 85-180 Hz
- **Female Range:** 165-255 Hz
- **Ambiguous Range:** 165-180 Hz (overlap zone)
- **Decision:** Based on median pitch over segment

### Combining (Stage 3)
- **Input:** Whisper segments + Diarization segments + Gender mapping
- **Method:** Time-based alignment and merging
- **Overlap Handling:** Longest overlapping speaker assignment
- **Output:** Unified timeline with speaker, text, gender

### LLM Analysis (Stage 4)
- **Provider:** Ollama (local)
- **Default Model:** llama3.2
- **Tasks:** Summarization, task extraction, key points
- **Context Window:** Full transcript
- **Output:** Text files + JSON

## 🔬 Performance Benchmarks

**Hardware:** RTX 3060 6GB, 16GB RAM, Intel Core i7

| Stage | Time (10min audio) | GPU Usage | Memory |
|-------|-------------------|-----------|---------|
| Stage 0: Audio | ~3s | 0% | 100MB |
| Stage 1: Whisper (medium) | ~120s | 90% | 2GB |
| Stage 2: Diarization | ~60s | 80% | 1.5GB |
| Stage 2.5: Gender | ~15s | 0% | 500MB |
| Stage 3: Combining | ~1s | 0% | 50MB |
| Stage 4: LLM | ~30s | 0%* | 4GB |
| **Total** | **~230s** | | |

*LLM chạy trên Ollama, có thể dùng GPU nếu cấu hình

## 📚 Dependencies Details

```
torch>=2.0.0          # Deep learning framework
pyannote.audio>=3.1.0 # Speaker diarization
faster-whisper>=0.8.0 # Fast transcription
librosa>=0.10.0       # Audio analysis
soundfile>=0.12.1     # Audio I/O
pydub>=0.25.1         # Audio conversion
scipy>=1.10.0         # Signal processing
ollama>=0.1.0         # LLM client
numpy>=1.24.0         # Numerical computing
huggingface-hub       # Model downloads
```

## 🎓 Workflow Examples

### Example 1: Process single audio
```bash
python main.py
```

### Example 2: Process with specific settings
```python
import config
config.AUDIO_FILE = "meeting.mp3"
config.WHISPER_MODEL = "large"
config.MIN_SPEAKERS = 2
config.MAX_SPEAKERS = 4
config.ENABLE_LLM_ANALYSIS = False
import main
main.main()
```

### Example 3: Batch processing
```python
import os
from src import audio_processor, whisper, diarization, gender_classifier, combiner

audio_files = [f for f in os.listdir("audios/") if f.endswith(".mp3")]

for audio_file in audio_files:
    print(f"\nProcessing {audio_file}...")
    audio_path = f"audios/{audio_file}"
    
    # Stage 0
    enhanced = audio_processor.enhance_audio(audio_path)
    
    # Stage 1-3
    whisper.transcribe_audio_optimized(enhanced)
    diarization.diarize_audio(enhanced)
    gender_classifier.classify_gender(enhanced)
    combiner.combine_results()
    
    print(f"✓ Done: {audio_file}")
```

### Example 4: Custom integration
```python
from src import combiner
import utils

# Load combining output
segments = utils.load_json("outputs/combining_output.json")

# Filter by speaker
speaker_00_text = [s['text'] for s in segments if s['speaker'] == 'SPEAKER_00']
print("SPEAKER_00:", ' '.join(speaker_00_text))

# Calculate speaking time
for speaker in ['SPEAKER_00', 'SPEAKER_01']:
    duration = sum(s['end'] - s['start'] for s in segments if s['speaker'] == speaker)
    print(f"{speaker}: {duration:.1f}s")
```

---

**Version:** 2.0  
**Last Updated:** November 2, 2025  
**Maintainer:** Internal Development Team  

📖 **Xem thêm:** [README.md](README.md) cho hướng dẫn đầy đủ
