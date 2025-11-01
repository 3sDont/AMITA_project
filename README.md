# Audio Processing Pipeline — Hướng dẫn chi tiết

Một hệ thống xử lý audio modular hỗ trợ hai pipeline chính: Standard (ổn định) và Optimized (tối ưu, nhanh). README này hướng dẫn cài đặt, sử dụng, cấu hình và phát triển.

---

## Mục lục
- Giới thiệu
- Cấu trúc dự án
- Cài đặt
- Hướng dẫn nhanh
  - Standard Pipeline
  - Optimized Pipeline (cache & parallel)
- Cấu hình
- Đầu ra, cache và logs
- Ví dụ thực tế
- Lời khuyên tối ưu hóa
- Phát triển & đóng góp
- Xử lý lỗi thường gặp
- License

---

## Giới thiệu
Hệ thống được thiết kế để:
- Tách tiếng, nhận diện số lượng và vị trí người nói.
- Hỗ trợ chế độ chạy chuẩn (ổn định) và chế độ tối ưu (dùng caching, xử lý song song, mô hình tối ưu).
- Dễ cấu hình qua file YAML, dễ mở rộng với module trong `src/core` và `src/pipeline`.

---

## Cấu trúc dự án
```
Paper for seminar/
├── src/
│   ├── core/              # Core processing modules
│   ├── pipeline/          # Standard & Optimized pipelines
│   ├── utils/             # Utilities
│   └── analysis/          # Analysis tools
├── scripts/               # Entry points
├── config/                # Configuration (e.g., default.yaml)
└── data/
    ├── input/
    ├── output/
    ├── cache/
    └── logs/
```

---

## Cài đặt

1. Tạo virtualenv (khuyến khích)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

2. Cài dependencies cho Standard:
```bash
pip install -r requirements.txt
```

3. Nếu muốn dùng pipeline tối ưu (faster models / GPU), cài thêm:
```bash
pip install -r requirements_optimized.txt
```

4. Thiết lập cấu hình mặc định (tùy chọn)
Sao chép `config/default.yaml.example` -> `config/default.yaml` và chỉnh tham số theo nhu cầu.

---

## Hướng dẫn nhanh

Chạy Standard Pipeline (ổn định):
```bash
python src/pipeline/standard_pipeline.py "data/input/audio.mp3" --min-speakers 2 --max-speakers 4
```

Chạy Optimized Pipeline (dùng cache, parallel):
- Lần đầu (với parallel và cache tạo mới):
```bash
python src/pipeline/optimized_pipeline.py "data/input/audio.mp3" --min-speakers 2 --max-speakers 4 --parallel
```
- Lần tiếp theo (tự động dùng cache nếu không thay đổi file):
```bash
python src/pipeline/optimized_pipeline.py "data/input/audio.mp3"
```
- Bắt buộc chạy lại, bỏ qua cache:
```bash
python src/pipeline/optimized_pipeline.py "data/input/audio.mp3" --force-rerun
```

Thông thường các option:
- --min-speakers N: số lượng speaker tối thiểu
- --max-speakers N: số lượng speaker tối đa
- --parallel: bật xử lý song song (Optimized)
- --force-rerun: bỏ cache và chạy lại từ đầu
- --output PATH: đường dẫn lưu kết quả (nếu hỗ trợ)

---

## Cấu hình (config/default.yaml)
Các tham số quan trọng (ví dụ):
- model: tên mô hình chuyển mã (ASR) hoặc đường dẫn
- sample_rate: tần số mẫu đầu vào
- cache:
  - enabled: true/false
  - dir: data/cache/
- logging:
  - level: INFO/DEBUG
  - dir: data/logs/

Chỉnh `config/default.yaml` để thay đổi hành vi mặc định.

---

## Đầu ra, cache & logs
- Kết quả chuyển mã, phân đoạn speaker, metadata lưu trong `data/output/`.
- File trung gian và cache (vector, embeddings, kết quả tạm) lưu trong `data/cache/`.
- Log runtime nằm trong `data/logs/` (một file/phiên hoặc theo ngày tùy cấu hình).

Lưu ý: chế độ Optimized tận dụng cache để giảm thời gian chạy, nhất là với audio lớn hoặc nhiều lần chạy cùng file.

---

## Ví dụ thực tế

1) Chạy full pipeline cho một file podcast:
```bash
python src/pipeline/optimized_pipeline.py "data/input/podcast_episode_01.mp3" --min-speakers 2 --max-speakers 6 --parallel
```
Kết quả:
- data/output/podcast_episode_01/transcript.json
- data/output/podcast_episode_01/segments.vtt
- data/output/podcast_episode_01/speakers.csv

2) Chỉ cần chạy bước phân đoạn speaker (nếu script hỗ trợ):
```bash
python src/scripts/run_speaker_diarization.py "data/input/audio.mp3" --out data/output/
```

---

## Lời khuyên tối ưu hóa
- Dùng GPU + faster model trong `requirements_optimized.txt` để tăng tốc (cần driver & CUDA đúng phiên bản).
- Kích thước batch và chunking: tách file lớn thành các đoạn nhỏ (~30-60s) để cân bằng bộ nhớ và độ chính xác.
- Sử dụng cache cho pipeline lặp lại cùng file để tiết kiệm thời gian.

---

## Phát triển & đóng góp
- Thêm module mới: đặt trong `src/core/` và expose qua `src/pipeline/*`.
- Tests: đặt test vào `tests/` (nếu có). Chạy `pytest` để kiểm tra.
- Pull request: mô tả rõ thay đổi, benchmark nếu có (tốc độ / bộ nhớ).

---

## Xử lý lỗi thường gặp
- "ModuleNotFoundError": kiểm tra virtualenv và pip install -r requirements.txt.
- GPU không dùng được: kiểm tra phiên bản CUDA, driver, và tương thích phiên bản package (torch, cuda toolkit).
- Cache không được cập nhật: dùng `--force-rerun` để bỏ cache hoặc xóa `data/cache/<filehash>/`.

---

## License
Kiểm tra file LICENSE ở thư mục gốc. Mặc định: MIT (hoặc chỉnh theo dự án).

---

Nếu cần, tôi có thể:
- Thêm ví dụ `config/default.yaml` mẫu.
- Viết hướng dẫn debug chi tiết hơn cho từng module.
- Viết script deploy / Dockerfile.
