# 📦 Installation Guide

Hướng dẫn cài đặt chi tiết cho Audio Transcription & Diarization Pipeline.

## 🔧 System Requirements

### Tối thiểu
- **OS:** Windows 10+, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python:** 3.8 - 3.11 (khuyến nghị 3.10)
- **RAM:** 8GB
- **Disk:** 10GB trống (cho models và dependencies)

### Khuyến nghị
- **GPU:** NVIDIA GPU với CUDA support (GTX 1060 6GB trở lên)
- **RAM:** 16GB+
- **Disk:** 20GB+ (SSD)
- **CUDA:** 11.8+ hoặc 12.x
- **cuDNN:** Tương ứng với CUDA version

## 📋 Pre-requisites

### 1. Cài đặt Python

**Windows:**
```powershell
# Download từ python.org hoặc dùng Microsoft Store
# Khuyến nghị: Python 3.10.x
winget install Python.Python.3.10
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

**macOS:**
```bash
brew install python@3.10
```

### 2. Cài đặt ffmpeg (Required)

**Windows:**
```powershell
# Cách 1: Dùng Chocolatey
choco install ffmpeg

# Cách 2: Dùng scoop
scoop install ffmpeg

# Cách 3: Download manual
# https://ffmpeg.org/download.html
# Thêm vào PATH
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

### 3. Cài đặt CUDA Toolkit (Optional - cho GPU)

**Windows/Linux:**
1. Download từ: https://developer.nvidia.com/cuda-downloads
2. Cài CUDA 11.8 hoặc 12.x
3. Cài cuDNN tương ứng
4. Thêm vào PATH

Verify:
```bash
nvidia-smi
nvcc --version
```

## 🚀 Installation Steps

### Step 1: Clone hoặc Download Project

```bash
# Nếu có git
git clone <repository-url>
cd PyannoteTest

# Hoặc download và giải nén ZIP
```

### Step 2: Tạo Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Nếu gặp lỗi execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows CMD:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Cài đặt PyTorch (GPU or CPU)

**GPU (CUDA 11.8):**
```bash
pip install torch==2.4.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch==2.4.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch==2.4.0+cpu torchaudio==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

Verify:
```python
python -c "import torch; print(torch.cuda.is_available())"
# True = GPU available, False = CPU only
```

### Step 5: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

Nếu gặp lỗi, cài từng package:
```bash
pip install pyannote.audio==3.1.1
pip install faster-whisper==0.10.0
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install pydub==0.25.1
pip install scipy==1.11.4
pip install ollama==0.1.7
pip install numpy==1.24.3
pip install huggingface-hub==0.20.0
```

### Step 6: Đăng ký HuggingFace Token

1. Truy cập: https://huggingface.co/settings/tokens
2. Tạo token mới (Read access)
3. Accept terms:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

4. Sửa `config.py`:
```python
HF_TOKEN = "hf_your_token_here"
```

### Step 7: Cài đặt Ollama (Optional)

**Windows:**
```powershell
# Download từ: https://ollama.ai/download
# Chạy installer và start service
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

Pull model:
```bash
ollama pull llama3.2
```

Verify:
```bash
ollama list
```

### Step 8: Test Installation

```bash
# Test imports
python -c "import torch; import pyannote.audio; import faster_whisper; print('✓ All imports OK')"

# Test GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Test Ollama (nếu cài)
ollama run llama3.2 "Hello"
```

## ✅ Verification

### Quick Test

```bash
# Activate environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# Test individual modules
python src/audio_processor.py
python src/whisper.py
python src/diarization.py
```

### Full Pipeline Test

```bash
# Đặt audio file trong config.py
python main.py
```

## 🐛 Common Issues

### Issue 1: "No module named 'torch'"
```
Solution:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: "RuntimeError: CUDA not available"
```
Check:
1. NVIDIA GPU driver installed?
2. CUDA toolkit installed?
3. PyTorch installed with CUDA?
   python -c "import torch; print(torch.version.cuda)"
```

### Issue 3: "FileNotFoundError: ffmpeg"
```
Solution:
- Windows: Add ffmpeg to PATH
- Linux: sudo apt install ffmpeg
- Mac: brew install ffmpeg
```

### Issue 4: "HTTPError 401: Unauthorized" (HuggingFace)
```
Solution:
1. Get token from https://huggingface.co/settings/tokens
2. Accept terms for pyannote models
3. Update HF_TOKEN in config.py
```

### Issue 5: "Connection refused" (Ollama)
```
Solution:
- Start Ollama service: ollama serve
- Or disable LLM: ENABLE_LLM_ANALYSIS = False in config.py
```

### Issue 6: "ImportError: DLL load failed" (Windows)
```
Solution:
Install Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Issue 7: Memory errors
```
Solution:
1. Close other applications
2. Use smaller Whisper model (tiny, base, small)
3. Set USE_GPU = False if GPU memory insufficient
```

## 🔄 Updating

```bash
# Activate environment
.venv\Scripts\Activate.ps1

# Update packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade pyannote.audio

# Check versions
pip list | grep torch
pip list | grep pyannote
pip list | grep faster-whisper
```

## 🗑️ Uninstallation

```bash
# Deactivate environment
deactivate

# Remove virtual environment
# Windows
Remove-Item -Recurse -Force .venv

# Linux/Mac
rm -rf .venv

# Optionally remove cache
Remove-Item -Recurse -Force outputs/
Remove-Item -Recurse -Force __pycache__/
```

## 📦 Package Versions (Tested)

```
Python: 3.10.11
torch: 2.4.0+cu121
torchaudio: 2.4.0+cu121
pyannote.audio: 3.1.1
faster-whisper: 0.10.0
librosa: 0.10.1
soundfile: 0.12.1
pydub: 0.25.1
scipy: 1.11.4
ollama: 0.1.7
numpy: 1.24.3
huggingface-hub: 0.20.0
```

## 🎓 Alternative Installation Methods

### Using Conda

```bash
conda create -n pyannote python=3.10
conda activate pyannote
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Using Docker (Advanced)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.10 python3-pip ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app
COPY . .
CMD ["python", "main.py"]
```

## 🆘 Getting Help

1. Check [README.md](README.md) for usage guide
2. Check [README_MODULES.md](README_MODULES.md) for technical details
3. Verify all dependencies installed: `pip list`
4. Check Python version: `python --version`
5. Check GPU: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

---

**Last Updated:** November 2, 2025  
**Python Version:** 3.8 - 3.11  
**Tested On:** Windows 11, Ubuntu 22.04, macOS 13
