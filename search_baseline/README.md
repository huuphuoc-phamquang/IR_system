# HCMAI24 Search: baseline

Hệ thống truy vấn video bằng câu mô tả, dùng trực tiếp output của notebook
[`keyframe_extraction_colab.ipynb`](../keyframe_extraction_colab.ipynb).

```
câu truy vấn (tiếng Anh) ─► CLIP text encoder ─► cosine với mọi keyframe ─► top-K ─► web UI / CSV nộp bài
```

| File | Việc |
|---|---|
| `build_index.py` | gộp `features-*/*.npy` và `map-keyframes/*.csv` thành 1 index, đồng thời ghi vị trí byte của từng ảnh trong file `.tar` (đọc ảnh không cần giải nén) |
| `engine.py` | tìm theo văn bản, tìm cảnh tương tự, truy vấn sự kiện A → B, keyframe lân cận |
| `app.py` + `static/index.html` | web UI: xem kết quả, xem lại video đúng thời điểm, lưu đáp án (GT), xuất CSV |
| `evaluate.py` | đo Recall@K và MRR trên bộ câu hỏi có đáp án, để so sánh các lần cải tiến |

Baseline tìm bằng phép nhân ma trận (brute force) trên GPU hoặc CPU. Vài trăm nghìn keyframe chỉ mất
khoảng vài chục ms, nên chưa cần Milvus hay FAISS.

## Chạy

```bash
pip install -r requirements.txt
```

1. Tải thư mục kết quả của notebook (`OUT_DIR`, ví dụ `HCMAI24_keyframes/`) từ Drive về máy.
2. Tạo index:

```bash
python build_index.py --data D:/data/HCMAI24_keyframes
```

Nhiều batch nằm ở các thư mục riêng thì đưa hết vào một lệnh, ra **một** index chung:

```bash
python build_index.py --data D:/4_Code/IR/Data/keyframes/Batch1 D:/4_Code/IR/Data/keyframes/Batch2 `
                      --out  D:/4_Code/IR/Data/keyframes/index
```

Video trùng tên giữa các thư mục thì lấy thư mục đầu tiên và in cảnh báo.

3. Mở web UI (thêm `--videos` để xem lại video; bỏ đi nếu không có video trên máy):

```bash
python app.py --index D:/data/HCMAI24_keyframes/index --videos D:/4_Code/IR/batch1
```

Sau đó mở http://localhost:8000. Trên server Ubuntu không có GPU, app tự chạy bằng CPU. Thêm `--host 0.0.0.0`
để máy khác trong mạng truy cập được.

## Nộp lên DRES

Mật khẩu nên để trong biến môi trường, đừng gõ vào dòng lệnh (nó vào history của shell):

```bash
export DRES_URL=https://dres-cua-ban-to-chuc    # Windows: set DRES_URL=...
export DRES_USER=team_cua_ban
export DRES_PASS=...
python app.py --index D:/data/HCMAI24_keyframes/index --videos D:/4_Code/IR/Data/batch1
```

Lúc khởi động app in ra `DRES: <url>`, và dòng info trên UI hiện "DRES đã nối". Mở chi tiết một
keyframe rồi bấm **Nộp lên DRES** — có hộp xác nhận vì nộp rồi không rút lại được. Không cấu hình
thì nút bị tắt, phần còn lại của app chạy bình thường.

`dres.py` lo 3 call: `login` → `client/evaluation/list` (lấy cái `ACTIVE`) → `submit/{id}`. `start`/`end`
tính bằng milliseconds **theo fps của chính video đó** (lấy từ `meta.csv`), vì batch2 trộn 25/30fps.
Toast hiện nguyên văn phản hồi của DRES; `412` thường là chưa có task đang chạy hoặc cần thêm `taskName`.

## Web UI
- **Văn bản**: mô tả cảnh **bằng tiếng Anh**. CLIP ViT-B-32 gần như không hiểu tiếng Việt.
- **Sự kiện A → B**: tìm keyframe khớp A mà trong vòng N giây sau đó, cùng video, có cảnh khớp B.
  Điểm = sim(A) + max sim(B). Ảnh nhỏ ở góc thẻ kết quả là cảnh B khớp nhất.
- **Tối đa mỗi video**: tránh để một video chiếm hết top-K.
- Bấm vào ảnh để mở chi tiết: xem video (tự tua tới trước keyframe 2 giây), keyframe lân cận, **Tìm cảnh
  tương tự** (dùng chính ảnh đó làm truy vấn), **Lưu làm GT**, Copy `video,frame`. Phím ← → để chuyển kết quả, Esc để đóng.
- **Xuất CSV**: top 100 dạng `L01_V001,12345` (video, frame_idx).

## Đánh giá: làm trước khi cải tiến

Không có bộ câu hỏi có đáp án thì không biết cải tiến nào thật sự có ích. Cách nhanh nhất để có bộ câu hỏi:
mở UI, gõ mô tả một cảnh bạn **đã biết trước** (xem trên video), tìm đúng cảnh đó rồi bấm **Lưu làm GT**.
Mỗi lần bấm sẽ ghi một dòng vào `queries/gt_ui.jsonl`. Nên có từ 50 câu trở lên, trải đều các video, và viết câu
**trước** khi tìm để không bị thiên vị theo kết quả. Nếu có đề và đáp án HCMAI24 chính thức thì chuyển sang
cùng định dạng:

```json
{"id": "q1", "query": "...", "video": "L01_V001", "start": 1200, "end": 1500}
```

`start`/`end` là khoảng `frame_idx` đúng. Thêm `"query_b"` và `"window"` cho câu hỏi dạng A → B.

```bash
python evaluate.py --index D:/data/HCMAI24_keyframes/index --queries queries/gt_ui.jsonl --out runs/baseline.csv
```

Kết quả gồm R@1/5/10/20/50/100 và MRR, tính ở 2 mức: đúng khung hình (lệch không quá `--tol` giây) và đúng video.
File `--out` ghi rank của từng câu, dùng để so sánh với các lần chạy sau.
`queries/sample_smoke.jsonl` chỉ là 10 câu kiểm tra code chạy trên 4 video L01_V001–V004, không phải bộ đánh giá thật.

## Hướng cải tiến (theo thứ tự nên thử)

| # | Cải tiến | Sửa ở đâu |
|---|---|---|
| 1 | Dịch câu hỏi tiếng Việt sang tiếng Anh (Google Translate / LLM) trước khi encode | `engine.text_scores` |
| 2 | Model mạnh hơn: ViT-H-14-378 (DFN), SigLIP2, MetaCLIP 2 (đa ngôn ngữ). Encode lại keyframe trên GPU, lưu vào `features-<model>/` | notebook embedding mới + `build_index.py --feat-dir --model --pretrained` |
| 3 | OCR chữ trên màn hình và ASR lời đọc → tìm bằng BM25, kết hợp với điểm CLIP (RRF) | engine mới `text_scores` hỗn hợp |
| 4 | Gom các keyframe liền kề trùng cảnh, giới hạn số kết quả mỗi đoạn | `_rank` |
| 5 | Phản hồi liên quan: chọn vài ảnh đúng → tìm theo trung bình embedding của chúng | `similar` |
| 6 | Rerank top 20–50 bằng VLM (Qwen2.5-VL) khi có GPU | sau `search` |
