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

## Lần đầu: lấy code và dựng index (SigLIP2, khuyên dùng)

SigLIP2 tốt hơn hẳn ViT-B-32 và hiểu thẳng tiếng Việt (chi tiết số đo ở
[`search_baseline/README.md`](search_baseline/README.md)). Không cần tự encode: lấy file
`siglip2_features.zip` (835 MB) của nhóm.

**1. Lấy code mới nhất**

```powershell
git clone https://github.com/huuphuoc-phamquang/IR_system.git   # lần đầu
cd IR_system
git pull                                                        # các lần sau
cd search_baseline
pip install -r requirements.txt
```

Kiểm tra torch nhận GPU — phải in `True`:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Nếu ra `+cpu False` thì cài lại bản CUDA. Xem số "CUDA Version" ở góc phải `nvidia-smi` rồi chọn
bản không lớn hơn số đó (`cu130` cho 13.x, `cu128` cho 12.8+, `cu126` cho 12.6+):

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**2. Đặt dữ liệu**

Giải nén `siglip2_features.zip` vào thư mục **đang chứa** `Batch1\` và `Batch2\` (gọi là `<X>`).
File tự rơi đúng chỗ:

```
<X>\Batch1\keyframes\                   *.tar   (đã có sẵn)
<X>\Batch1\map-keyframes\               *.csv   (đã có sẵn)
<X>\Batch1\features-siglip2-l16-384\    *.npy   <- từ zip
<X>\Batch2\...                          (tương tự)
```

`keyframes\` và `map-keyframes\` phải là **cùng bản trích keyframe** với nhóm; bản khác thì số
keyframe lệch và video đó bị bỏ qua.

**3. Dựng index** (khoảng 2 phút)

```powershell
python build_index.py --data <X>/Batch1 <X>/Batch2 --feat-dir features-siglip2-l16-384 `
                      --model ViT-L-16-SigLIP2-384 --pretrained webli --out <X>/index_siglip2
```

Phải ra `427,601 keyframe, 726 video, dim 1024` và **không** có dòng `!! bo qua`.

Không chép thẳng thư mục index từ máy khác: `meta.csv` ghi đường dẫn tuyệt đối tới file `.tar`,
sang máy khác sẽ lỗi ảnh. Luôn dựng lại bằng lệnh trên.

**4. Chạy thử**

```powershell
python app.py --index <X>/index_siglip2 --videos <thư mục chứa .mp4> --port 8124
```

Mở http://localhost:8124, dòng info phải ghi `427,601 keyframe · 726 video · ViT-L-16-SigLIP2-384`.
Lần chạy đầu tải weight SigLIP2 (vài GB). Bỏ `--videos` nếu không có video gốc — vẫn tìm được,
chỉ không xem lại được clip.

### Bộ cũ ViT-B-32

Vẫn dùng được, dựng bằng features gốc của notebook:

```powershell
python build_index.py --data <X>/Batch1 <X>/Batch2 --out <X>/index
```

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
python app.py --index D:/4_Code/IR/Data/keyframes/index_siglip2 --videos D:/4_Code/IR/Data --port 8123
```

Mở http://localhost:8123 (thêm `--host 0.0.0.0` để cả nhóm dùng chung một server).
Muốn quay về bộ cũ thì đổi `--index` thành `.../keyframes/index`.
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
