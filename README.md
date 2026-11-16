# IR_system — HCMAI24 interactive video retrieval

Tìm cảnh trong ~225 giờ video bằng câu mô tả tiếng Anh, ảnh, hoặc clip; xem lại đúng thời điểm
và nộp thẳng lên DRES.

```
video .mp4 ──► keyframe_extraction_colab.ipynb (GPU, Colab) ──► keyframes/*.tar
                                                                map-keyframes/*.csv
                                                                features-vit-b-32/*.npy
                                                                        │
                                              search_baseline/build_index.py
                                                                        ▼
                                           index (features.npy + meta.csv) ──► app.py (web UI)
```

| Thư mục | Nội dung |
|---|---|
| [`search_baseline/`](search_baseline/README.md) | Hệ thống tìm kiếm + web UI + nộp DRES. **Xem README trong đó để chạy.** |
| `keyframe_extraction_colab.ipynb` | Trích keyframe + embedding CLIP trên Colab |
| `explore_zip_structure.ipynb` | Xem cấu trúc file zip dữ liệu mà không cần giải nén |

Dữ liệu (video, keyframe, index) không nằm trong repo — xem `.gitignore`.

## Lần đầu: dựng index

```powershell
cd search_baseline
pip install -r requirements.txt
python build_index.py --data D:/4_Code/IR/Data/keyframes/Batch1 D:/4_Code/IR/Data/keyframes/Batch2 `
                      --out  D:/4_Code/IR/Data/keyframes/index
```

Index hiện tại: 427.601 keyframe / 726 video (L01–L24), ViT-B-32, tìm ~21 ms trên CPU.

## Mỗi buổi thi: chạy với DRES

Mở **hai** cửa sổ PowerShell, cả hai đều cần 3 biến môi trường (đặt trong chính cửa sổ đó, mở
tab mới là mất). Đừng truyền mật khẩu qua `--dres-pass`: nó vào `Get-History` và file history.

```powershell
$env:DRES_URL  = "https://aiclub.uit.edu.vn/dres"   # phần trước /evaluation/... trên thanh địa chỉ
$env:DRES_USER = "team_cua_ban"
$env:DRES_PASS = "..."
```

**Cửa sổ 1 — web UI + nộp bài:**

```powershell
cd D:\4_Code\IR\search_baseline
python app.py --index D:/4_Code/IR/Data/keyframes/index --videos D:/4_Code/IR/Data --port 8123
```

Mở http://localhost:8123 (thêm `--host 0.0.0.0` để cả nhóm dùng chung một server).
Khởi động phải in `DRES: https://...`; nếu in `chua cau hinh` thì biến môi trường chưa tới được tiến trình.

**Cửa sổ 2 — tự bắt đề bài:**

```powershell
cd D:\4_Code\IR\search_baseline
python capture.py
```

Hỏi DRES mỗi 5 giây; có task mới thì lưu `runs/tasks/<tên task>/`:

| Loại task | File lưu ra | Dùng thế nào |
|---|---|---|
| `tkis` (3 câu mô tả) | `query.txt`, `hint.json` | dịch sang tiếng Anh, điền vào 3 ô **Chuỗi sự kiện A → B → C** |
| `vkis` (clip mục tiêu) | `00.mp4` | kéo thả file đó vào trang, hoặc `Ctrl+V` ảnh chụp màn hình |

Kiểm tra nhanh không cần mạng: `python dres.py`, `python capture.py --selftest`.
