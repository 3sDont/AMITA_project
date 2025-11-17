# 🎮 HƯỚNG DẪN BẬT/TẮT GPU

## 📋 Tóm tắt

Hệ thống hỗ trợ chạy trên **GPU (CUDA)** hoặc **CPU**. GPU nhanh hơn 5-10x nhưng yêu cầu NVIDIA GPU.

---

## ⚙️ Cách bật/tắt GPU

### 1️⃣ Mở file `config.py`

Tìm dòng cấu hình GPU:

```python
# ====================== GPU SETTINGS ======================
USE_GPU = True  # True: Dùng GPU nếu có | False: Force dùng CPU
```

### 2️⃣ Thay đổi giá trị

| Chế độ | Cài đặt | Khi nào dùng |
|--------|---------|--------------|
| **GPU** | `USE_GPU = True` | Có NVIDIA GPU, muốn nhanh |
| **CPU** | `USE_GPU = False` | Không có GPU, hoặc muốn tiết kiệm điện |
| **Auto** | `USE_GPU = True` | Để hệ thống tự chọn (khuyến nghị) |

### 3️⃣ Lưu và chạy

```bash
python main.py
```

---

## 🔍 Kiểm tra GPU

### Check GPU có sẵn không:

```bash
python check_gpu.py
```

Kết quả sẽ hiển thị:
- ✅ GPU có sẵn → Tên GPU, VRAM, CUDA version
- ❌ GPU không có → Hướng dẫn cài đặt

### Test toggle GPU:

```bash
python test_gpu_toggle.py
```

---

## ⚡ So sánh tốc độ

**Audio 3 phút, 4 speakers:**

| Thành phần | GPU (RTX 3060) | CPU (i7-11800H) | Tăng tốc |
|------------|----------------|-----------------|----------|
| Whisper medium | ~30-60s | ~5-10 phút | **5-10x** |
| Pyannote diarization | ~10-20s | ~1-3 phút | **3-6x** |
| **Tổng** | **~1 phút** | **~8 phút** | **~8x** |

---

## 📊 Theo dõi GPU usage

### Trong khi chạy pipeline:

Mở terminal mới và chạy:

```bash
nvidia-smi -l 1
```

Sẽ hiển thị:
- GPU usage (%)
- VRAM usage (MB)
- Temperature (°C)
- Power usage (W)

### Ví dụ output:

```
+-----------------------------------------------------------------------------------------+
| GPU   Name                  Memory-Usage | GPU-Util  Compute M. |
|=========================================+=======================|
|   0  NVIDIA GeForce RTX 3060     3500MiB |    95%      Default  |
+-----------------------------------------------------------------------------------------+
```

---

## 🛠️ Troubleshooting

### ❌ GPU không được nhận diện

**Nguyên nhân:**
- Chưa cài NVIDIA driver
- Chưa cài CUDA toolkit
- PyTorch phiên bản CPU-only

**Giải pháp:**

1. **Cài NVIDIA driver:**
   - Download: https://www.nvidia.com/drivers
   - Hoặc dùng GeForce Experience

2. **Cài PyTorch GPU:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify:**
   ```bash
   python check_gpu.py
   ```

### ⚠️ Out of Memory (OOM)

**Triệu chứng:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**

1. **Đóng các app khác đang dùng GPU:**
   - Game
   - Video editing
   - Other ML models

2. **Dùng model nhỏ hơn:**
   ```python
   # Trong config.py
   WHISPER_MODEL = "small"  # Thay vì "medium" hoặc "large"
   ```

3. **Force dùng CPU:**
   ```python
   USE_GPU = False
   ```

4. **Clear cache thủ công:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### 🐌 GPU chậm hơn CPU?

Điều này BẤT THƯỜNG. Kiểm tra:

1. **GPU đang bị throttle?**
   ```bash
   nvidia-smi
   # Check nhiệt độ > 80°C?
   # Check Power Limit?
   ```

2. **Audio quá ngắn?**
   - GPU có overhead, chỉ nhanh với audio dài (>1 phút)

3. **Laptop mode?**
   - Bật Performance Mode trong Windows
   - Cắm sạc điện

---

## 💡 Tips tối ưu

### 1. Sử dụng GPU hiệu quả

```python
# ✅ Tốt: Load model 1 lần, xử lý nhiều file
model = whisper.load_model("medium", device="cuda")
for audio_file in audio_files:
    model.transcribe(audio_file)

# ❌ Chậm: Load model mỗi lần
for audio_file in audio_files:
    model = whisper.load_model("medium", device="cuda")  # Lãng phí!
    model.transcribe(audio_file)
```

### 2. Batch processing

Nếu có nhiều file, xử lý theo batch:

```bash
# Chạy cùng lúc nhiều file
python main.py  # File 1
# GPU sẽ warm, file tiếp theo nhanh hơn
```

### 3. Model size vs Accuracy vs Speed

| Model | Size | VRAM | CPU RAM | Tốc độ GPU | Độ chính xác |
|-------|------|------|---------|------------|--------------|
| tiny | 39M | ~1GB | ~1GB | Rất nhanh | Thấp |
| base | 74M | ~1GB | ~1GB | Nhanh | Trung bình |
| small | 244M | ~2GB | ~2GB | Khá nhanh | Tốt |
| medium | 769M | ~5GB | ~5GB | Vừa | Rất tốt ✅ |
| large | 1550M | ~10GB | ~10GB | Chậm | Tốt nhất |

**Khuyến nghị:**
- RTX 3060 (6GB VRAM) → **medium** ✅
- RTX 3050 (4GB VRAM) → small
- RTX 3090 (24GB VRAM) → large

---

## 📝 Log output mẫu

### Khi USE_GPU = True và có GPU:

```
🎮 Sử dụng GPU: NVIDIA GeForce RTX 3060 Laptop GPU
   VRAM: 6.00 GB

⏳ Đang load Whisper model 'medium' on CUDA...
⏳ Đang transcribe audio...
✅ Hoàn thành!
   - Device: CUDA
```

### Khi USE_GPU = False:

```
🖥️  Sử dụng CPU (USE_GPU=False)

⏳ Đang load Whisper model 'medium' on CPU...
⏳ Đang transcribe audio...
✅ Hoàn thành!
   - Device: CPU
```

### Khi USE_GPU = True nhưng không có GPU:

```
⚠️  GPU không khả dụng, fallback về CPU

⏳ Đang load Whisper model 'medium' on CPU...
⏳ Đang transcribe audio...
✅ Hoàn thành!
   - Device: CPU
```

---

## 🔗 Tài liệu tham khảo

- [PyTorch CUDA](https://pytorch.org/docs/stable/cuda.html)
- [Whisper GitHub](https://github.com/openai/whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

---

## ❓ FAQ

**Q: Tôi có GPU AMD, có dùng được không?**  
A: Không. Whisper và Pyannote chỉ hỗ trợ NVIDIA GPU (CUDA). GPU AMD cần ROCm nhưng không tương thích.

**Q: Laptop của tôi có 2 GPU (Intel iGPU + NVIDIA), dùng cái nào?**  
A: PyTorch tự động dùng NVIDIA GPU nếu có driver.

**Q: Có thể dùng Google Colab GPU miễn phí không?**  
A: Có! Upload code lên Colab, chọn Runtime → Change runtime type → GPU.

**Q: MacBook M1/M2 có dùng được GPU không?**  
A: Có, nhưng dùng MPS (Metal Performance Shaders), không phải CUDA. Cần cài PyTorch cho Apple Silicon.

---

**📌 Tóm lại:** Đặt `USE_GPU = True` trong `config.py` để dùng GPU. Hệ thống tự động fallback về CPU nếu không có GPU!
