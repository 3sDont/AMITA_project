# 🐍 HƯỚNG DẪN KÍCH HOẠT VIRTUAL ENVIRONMENT (venv)

## 📍 Vị trí venv của bạn
```
D:\File\Seminar\PyannoteTest\.venv\
```

---

## ✅ Cách kích hoạt venv

### **Windows PowerShell** (Bạn đang dùng)

#### **Cách 1: Script Activate**
```powershell
# Di chuyển đến thư mục project
cd D:\File\Seminar\PyannoteTest

# Kích hoạt venv
.\.venv\Scripts\Activate.ps1
```

#### **Cách 2: Chỉ định đường dẫn đầy đủ**
```powershell
D:\File\Seminar\PyannoteTest\.venv\Scripts\Activate.ps1
```

#### **Cách 3: Dùng activate.bat (nếu ps1 bị chặn)**
```powershell
.\.venv\Scripts\activate.bat
```

---

### **Nếu gặp lỗi "running scripts is disabled"**

PowerShell có thể chặn script vì chính sách bảo mật.

#### **Fix tạm thời (chỉ session hiện tại):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

#### **Fix vĩnh viễn (cần quyền Admin):**
```powershell
# Mở PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Sau đó kích hoạt venv bình thường
cd D:\File\Seminar\PyannoteTest
.\.venv\Scripts\Activate.ps1
```

---

## ✅ Kiểm tra venv đã kích hoạt

Sau khi activate thành công, bạn sẽ thấy:

```powershell
(.venv) PS D:\File\Seminar\PyannoteTest>
       ^^^^^^ Dấu hiệu venv đã bật
```

Hoặc kiểm tra Python path:
```powershell
which python  # hoặc
Get-Command python
```

Kết quả phải là:
```
D:\File\Seminar\PyannoteTest\.venv\Scripts\python.exe
```

---

## 🚀 Chạy code sau khi activate venv

```powershell
# 1. Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# 2. Chạy pipeline
python main.py

# hoặc chạy từng bước
python step1_whisper.py
python step2_diarization.py
# ...
```

---

## 🔄 Tắt venv

```powershell
deactivate
```

---

## 💡 Alternative: Chạy trực tiếp KHÔNG CẦN activate

Nếu không muốn activate, chạy trực tiếp với đường dẫn đầy đủ:

```powershell
# Windows
D:\File\Seminar\PyannoteTest\.venv\Scripts\python.exe main.py

# Hoặc dùng biến
$python = "D:\File\Seminar\PyannoteTest\.venv\Scripts\python.exe"
& $python main.py
```

---

## 📋 Quick Commands

### **Setup lần đầu:**
```powershell
cd D:\File\Seminar\PyannoteTest
.\.venv\Scripts\Activate.ps1
pip install -r requirements_updated.txt
```

### **Chạy pipeline:**
```powershell
cd D:\File\Seminar\PyannoteTest
.\.venv\Scripts\Activate.ps1
python main.py
```

### **Kiểm tra packages:**
```powershell
.\.venv\Scripts\Activate.ps1
pip list
```

---

## 🐛 Troubleshooting

### **Lỗi: "Activate.ps1 cannot be loaded"**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### **Lỗi: "python not found" sau activate**
```powershell
# Check venv path
ls .\.venv\Scripts\

# Recreate venv nếu cần
python -m venv .venv --clear
```

### **Lỗi: Import module failed**
```powershell
# Reinstall packages
.\.venv\Scripts\Activate.ps1
pip install -r requirements_updated.txt
```

---

## 🎯 VS Code Integration

Nếu dùng VS Code, có thể chọn Python interpreter:

1. **Ctrl + Shift + P** → "Python: Select Interpreter"
2. Chọn: `.venv\Scripts\python.exe`
3. VS Code tự động activate venv khi mở terminal

---

## 📝 Script tự động (tạo file `run.ps1`)

```powershell
# Tạo file run.ps1
@"
# Auto activate venv and run main.py
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\\.venv\\Scripts\\Activate.ps1
python main.py
deactivate
"@ | Out-File -FilePath run.ps1 -Encoding UTF8

# Chạy
.\run.ps1
```

---

## ✅ Checklist

- [ ] Mở PowerShell
- [ ] `cd D:\File\Seminar\PyannoteTest`
- [ ] `.\.venv\Scripts\Activate.ps1`
- [ ] Thấy `(.venv)` ở đầu prompt
- [ ] `python main.py`

---

## 🎉 Tóm tắt nhanh

```powershell
# Bước 1: Đến thư mục
cd D:\File\Seminar\PyannoteTest

# Bước 2: Activate venv
.\.venv\Scripts\Activate.ps1

# Bước 3: Chạy code
python main.py

# Bước 4: Tắt venv (khi xong)
deactivate
```

---

**Lưu ý:** Luôn activate venv trước khi chạy code để dùng đúng packages đã cài!
