# HCMAI24 Search: baseline

Hệ thống truy vấn video bằng câu mô tả, dùng trực tiếp output của notebook
[`keyframe_extraction_colab.ipynb`](../keyframe_extraction_colab.ipynb).

```
câu truy vấn (tiếng Anh) ─► CLIP text encoder ─► cosine với mọi keyframe ─► top-K ─► web UI / CSV nộp bài
```

| File | Việc |
|---|---|
| `build_index.py` | gộp `features-*/*.npy` và `map-keyframes/*.csv` thành 1 index, đồng thời ghi vị trí byte của từng ảnh trong file `.tar` (đọc ảnh không cần giải nén) |
| `engine.py` | tìm theo văn bản/ảnh/clip, tìm cảnh tương tự, chuỗi sự kiện A → B → C, keyframe lân cận |
| `app.py` + `static/index.html` | web UI: xem kết quả, xem lại video đúng thời điểm, lưu đáp án (GT), xuất CSV |
| `dres.py` | nộp kết quả lên DRES: `login` → evaluation `ACTIVE` → `submit` |
| `capture.py` | hỏi DRES mỗi 5s, có task mới thì lưu đề bài (text + clip) vào `runs/tasks/` |
| `video.py` | lấy khung hình cách đều từ một clip, để tìm bằng cả clip thay vì 1 ảnh |
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
python app.py --index D:/data/HCMAI24_keyframes/index --videos D:/4_Code/IR/Data
```

Sau đó mở http://localhost:8000. Trên server Ubuntu không có GPU, app tự chạy bằng CPU. Thêm `--host 0.0.0.0`
để máy khác trong mạng truy cập được.

## Nộp lên DRES

Mật khẩu nên để trong biến môi trường, đừng gõ vào dòng lệnh (nó vào history của shell):

```powershell
# PowerShell -- `set VAR=x` KHONG tao bien moi truong, phai dung $env:
$env:DRES_URL  = "https://aiclub.uit.edu.vn/dres"   # phần trước /evaluation/... trên thanh địa chỉ
$env:DRES_USER = "team_cua_ban"
$env:DRES_PASS = "..."
python app.py --index D:/4_Code/IR/Data/keyframes/index --videos D:/4_Code/IR/Data --port 8123
```

```bash
# Linux / macOS
export DRES_URL=https://aiclub.uit.edu.vn/dres
export DRES_USER=team_cua_ban
export DRES_PASS=...
```

Tự bắt đề bài (cửa sổ riêng, cùng 3 biến trên): `python capture.py` — xem mục "Bắt đề bài" bên dưới.

Lúc khởi động app in ra `DRES: <url>`, và dòng info trên UI hiện "DRES đã nối". Mở chi tiết một
keyframe rồi bấm **Nộp lên DRES** — có hộp xác nhận vì nộp rồi không rút lại được. Không cấu hình
thì nút bị tắt, phần còn lại của app chạy bình thường.

`dres.py` lo 3 call: `login` → `client/evaluation/list` (lấy cái `ACTIVE`) → `submit/{id}`. `start`/`end`
tính bằng milliseconds **theo fps của chính video đó** (lấy từ `meta.csv`), vì batch2 trộn 25/30fps.
Kết quả chấm hiện cạnh nút và trên thanh trạng thái, **không tự biến mất**: `✅ ĐÚNG` / `❌ SAI` /
`⏳ CHƯA CHẤM` (HTTP 202) / `⚠️ KHÔNG CHẤM ĐƯỢC`, kèm `video @ frame` đã nộp. `412` thường là chưa có
task đang chạy hoặc cần thêm `taskName`.

## Bắt đề bài

```powershell
python capture.py            # hoi DRES moi 5s; --interval, --out de doi
python capture.py --selftest # kiem tra code, khong can mang
```

Task mới thì lưu vào `runs/tasks/<tên task>/`:

| Loại | File | Dùng thế nào |
|---|---|---|
| `tkis` | `query.txt` (3 câu), `hint.json` | dịch sang tiếng Anh → 3 ô **Chuỗi sự kiện A → B → C** |
| `vkis` | `00.mp4` (clip mục tiêu) | kéo thả file vào trang, hoặc `Ctrl+V` ảnh chụp màn hình |
| `qakis` | `query.txt` (3 câu, kết thúc bằng "Đáp án của câu hỏi là gì?") | tìm cảnh như tkis, **đọc đáp án trên màn hình**, rồi điền vào ô đáp án trong drawer |

### qakis: nộp chữ, không phải vị trí video

Mở chi tiết một keyframe, ô dưới nút Nộp đã điền sẵn mẫu `<đáp án>-<video>-<ms>`. Thay `<đáp án>`
bằng đáp án thật rồi bấm **Nộp lên DRES** — nó gửi `{"text": "..."}` thay vì `mediaItemName`.
Để trống ô đó (hoặc bấm ↺ rồi không sửa) thì nộp như bình thường. Nút ↺ điền lại mẫu theo keyframe đang xem.

> Hai nguồn của ban tổ chức **không khớp** về format: `submit_dres.js` dùng `<đáp án>-<video>-<ms>`,
> còn `dres_submit_demo.py` dùng `QA-<đáp án>-<video>-<ms>`. Vì vậy ô này cho sửa tự do — xác nhận
> format với ban tổ chức trước vòng thi thật.

Chỉ gọi `GET`, không nộp gì. Dùng 2 endpoint không có trong `oas-client.json` (lấy từ API đầy đủ mà
trang viewer dùng): `/evaluation/{id}/state` và `/evaluation/{id}/template/task/{tid}/hint`.

## Web UI
- **Văn bản**: mô tả cảnh **bằng tiếng Anh**. CLIP ViT-B-32 gần như không hiểu tiếng Việt.
- **Dịch Việt→Anh**: tích ô này thì câu tiếng Việt được dịch (Google Translate) rồi mới encode, và
  ô nhập bị ghi đè bằng bản tiếng Anh để bạn thấy CLIP thực sự nhận câu nào.
- **Chuỗi sự kiện A → B → C**: mỗi sự kiện sau phải nằm trong N giây sau sự kiện trước, cùng video.
  Điểm = tổng sim. Đúng dạng của task `tkis` (mở đầu / sau đó / cuối cùng). Bỏ trống C thì thành A → B.
  Ảnh nhỏ ở góc thẻ kết quả là cảnh B khớp nhất.
- **Tìm bằng ảnh/video**: nút **Tải ảnh/video…**, kéo thả file vào trang, hoặc `Ctrl+V` ảnh chụp màn hình.
  Video: lấy **2 khung/giây**, tối đa 30 khung — clip dài quá 15s thì tự giãn mật độ ra (30 khung
  trải đều cả clip). Mỗi khung so với toàn bộ 427.601 keyframe, điểm = **max** trên các khung
  (clip nhiều cảnh, khớp 1 cảnh là đủ). Encode theo lô nên 30 khung mất ~240 ms (GPU) / ~1,0 s (CPU).
- **Tối đa mỗi video**: tránh để một video chiếm hết top-K.
- Bấm vào ảnh để mở chi tiết: xem video (tự tua tới trước keyframe 2 giây), keyframe lân cận, **Tìm cảnh
  tương tự** (dùng chính ảnh đó làm truy vấn), **Lưu làm GT**, Copy `video,frame`. Phím ← → để chuyển kết quả, Esc để đóng.
- **Xuất CSV**: top 100 dạng `L01_V001,12345` (video, frame_idx).

## Bộ GT từ vòng thi thử

`capture.py` lưu cả clip **đáp án** (`target_*.mp4`) sau khi mỗi task kết thúc. `collect_gt.py`
dùng chính engine để tìm clip đáp án đó trong index -> ra `video, frame_idx`, tức một câu GT
hoàn chỉnh mà không phải gõ tay.

```powershell
python collect_gt.py --index D:/4_Code/IR/Data/keyframes/index          # ca kho -> tong bo GT
python collect_gt.py --index ... --tasks runs/tasks --out runs/gt_new   # chi task moi
```

| Thư mục | Gì ở trong |
|---|---|
| `runs/tasks/` | task mới, `capture.py` còn đang theo dõi (chưa có đáp án) |
| `runs/tasks_done/` | task đã xong, có `target_*.mp4` |
| `runs/gt/` | `gt.xlsx`, `gt.jsonl`, `summary.csv`, `skip.txt`, ảnh keyframe đáp án |

`gt.xlsx` là bản để soát bằng mắt: mỗi câu một dòng, **ảnh keyframe nhúng thẳng trong ô**, kèm
3 câu tiếng Việt và 3 câu tiếng Anh cạnh nhau. Cột **SAI?** để trống cho bạn đánh dấu; sau đó
chép cột `id` của những dòng đó vào `skip.txt` rồi chạy lại.

Vòng thi thử chạy lại cùng một đề nhiều lần; mỗi lần DRES cấp `taskId` mới nên `capture.py`
tạo thêm thư mục. `collect_gt.py` **gom theo `taskId` trong `hint.json`** (id của template, ổn định)
và giữ bản tốt nhất — ưu tiên bản có clip đáp án, rồi nhiều câu hint hơn, rồi mới nhất.
Thực tế: 71 thư mục trên đĩa → 35 task thật.

**Phải liếc qua các ảnh `.jpg`.** Điểm cao không đảm bảo đúng: clip đáp án hay mở đầu bằng cảnh
studio/MC dùng chung cho mọi bản tin, khớp 0.96 với hàng trăm bản tin khác. Đã thử lọc tự động
bằng margin (top1 trừ video khác tốt nhất) — **không tách được** cái sai khỏi cái đúng
(sai +0.0117 vs đúng +0.0186). Thấy ảnh nào sai thì thêm id vào `runs/gt/skip.txt`, chạy lại
sẽ bỏ qua. **Nên ghi `taskId` của template** (cột `tid` trong `summary.csv`) chứ không phải tên
thư mục — tên thư mục đổi mỗi lần đề chạy lại, câu sai sẽ lẻn về dưới id khác.

Số đo nền (10 câu, tol 3s, ViT-B-32, index 427k keyframe):

| | R@1 | R@5 | R@20 | MRR |
|---|---|---|---|---|
| Tiếng Việt · video đúng | 0.000 | 0.000 | 0.000 | 0.000 |
| **Tiếng Anh · video đúng** | **0.600** | 0.600 | 0.600 | 0.600 |
| Tiếng Anh · khung đúng (±3s) | 0.400 | 0.500 | 0.600 | 0.455 |

Dịch là từ **0% lên 60%**. Với ViT-B-32 thì tiếng Việt không phải "kém" mà là bằng không.

**vkis** (`gt_vkis.jsonl`, 8 câu) đo riêng vì không có chữ: truy vấn bằng clip đề bài `00.mp4`,
đáp án giải từ `target_*.mp4` — hai file khác nhau nên không vòng vo.

| | R@1 | MRR |
|---|---|---|
| vkis · tìm bằng clip · khung đúng | **1.000** | 1.000 |

```bash
python evaluate.py --index <index> --queries runs/gt/gt_vkis.jsonl --tol 3
```

Tức khi đáp án nằm trong dữ liệu, tìm bằng clip **không trượt câu nào**. Nút thắt của vkis là
đáp án có nằm trong bộ hay không, không phải chất lượng tìm kiếm.

`start`/`end` của GT lấy bằng **độ dài clip đáp án** quanh khung khớp nhất, không phải một khung:
khung khớp nhất với clip thường rơi vào cảnh nổi bật (câu B hoặc C), trong khi truy vấn chuỗi
trả về khung của câu A — cách nhau hàng chục giây. Lấy một khung thì "khung đúng" ra 0.000 dù
hệ thống tìm đúng.

## Tạo thêm GT bằng clip cắt ngẫu nhiên

Vòng thi thử chỉ cho ~10 câu tkis. Muốn thêm thì cắt clip ngẫu nhiên từ chính bộ dữ liệu rồi
nhờ công cụ khác (NotebookLM, Gemini…) **viết mô tả** — như vậy câu hỏi không bị thiên lệch theo
chỗ mạnh của CLIP, điều sẽ xảy ra nếu người làm hệ thống tự viết.

```powershell
python cut_clips.py                                  # 10 clip 30s -> runs/gt_clips/
#   dua clipNN.mp4 (hoac clipNN_sheet.jpg neu cong cu khong nhan video)
#   + prompt_notebooklm.md  -> dan ket qua vao runs/gt_clips/mota.txt
python descriptions_to_gt.py --text runs/gt_clips/mota.txt
python evaluate.py --index <index> --queries runs/gt_clips/gt_clips.jsonl --tol 3
```

Clip đặt tên `clip01…clip10` chứ không theo tên video, đáp án để riêng trong `answers.csv` —
công cụ viết mô tả không biết đáp án, và lúc bạn tự test cũng không bị lộ.

Mỗi clip phải qua kiểm tra: tìm bằng chính nó ra đúng video nguồn, score ≥ 0.90 và ít nhất
4/10 kết quả đầu cùng video. Bước này loại đoạn toàn cảnh studio/MC — thứ khớp 0.96 với hàng
trăm bản tin khác và từng làm hỏng một câu GT.

Mô tả sinh ra là **tiếng Việt**, nhớ bật ô "Dịch Việt→Anh" hoặc dịch trước khi đo.

## Đánh giá: làm trước khi cải tiến

Không có bộ câu hỏi có đáp án thì không biết cải tiến nào thật sự có ích. Cách nhanh nhất để có bộ câu hỏi:
mở UI, gõ mô tả một cảnh bạn **đã biết trước** (xem trên video), tìm đúng cảnh đó rồi bấm **Lưu làm GT**.
Mỗi lần bấm sẽ ghi một dòng vào `queries/gt_ui.jsonl`. Nên có từ 50 câu trở lên, trải đều các video, và viết câu
**trước** khi tìm để không bị thiên vị theo kết quả. Nếu có đề và đáp án HCMAI24 chính thức thì chuyển sang
cùng định dạng:

```json
{"id": "q1", "query": "...", "video": "L01_V001", "start": 1200, "end": 1500}
```

`start`/`end` là khoảng `frame_idx` đúng. Thêm `"query_b"`, `"query_c"` và `"window"` cho câu hỏi
dạng chuỗi sự kiện. `frame_idx` phải là khung hình trong video gốc, **không phải** cột `n` hay `pts_time`.

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
