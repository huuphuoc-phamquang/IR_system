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

## Chạy nhanh

```bash
cd search_baseline
pip install -r requirements.txt
python build_index.py --data <OUT_DIR>/Batch1 <OUT_DIR>/Batch2 --out <OUT_DIR>/index
python app.py --index <OUT_DIR>/index --videos <thu muc .mp4> --port 8123
```

Index hiện tại: 427.601 keyframe / 726 video (L01–L24), ViT-B-32, tìm ~21 ms trên CPU.
