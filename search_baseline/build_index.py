"""Gop output cua notebook trich keyframe thanh 1 index de truy van.

Dau vao (thu muc OUT_DIR cua notebook keyframe_extraction_colab.ipynb):
    keyframes/<vid>.tar            anh keyframe <vid>/<frame_idx:06d>.jpg
    map-keyframes/<vid>.csv        n, pts_time, fps, frame_idx
    features-vit-b-32/<vid>.npy    embedding CLIP (float16, L2-norm), cung thu tu voi csv

Dau ra (--out, mac dinh <data>/index):
    features.npy   [N, D] float16 -- tat ca keyframe, sap theo (video, frame_idx)
    meta.csv       id, video, n, frame_idx, pts_time, fps, tar, offset, size
                   (offset/size = vi tri byte cua anh trong file tar -> doc anh khong can giai nen)
    info.json      model, so keyframe, so video, ...

Vi du:
    python build_index.py --data D:/data/HCMAI24_keyframes
    python build_index.py --data .../Batch1 .../Batch2 --out .../index   # gop nhieu batch
"""
import argparse
import csv
import json
import os
import tarfile
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, nargs='+',
                   help='OUT_DIR cua notebook trich keyframe; dua nhieu thu muc -> gop thanh 1 index')
    p.add_argument('--out', default=None, help='thu muc index (mac dinh <data dau tien>/index)')
    p.add_argument('--feat-dir', default='features-vit-b-32', help='thu muc embedding trong --data')
    p.add_argument('--model', default='ViT-B-32', help='model open_clip da tao embedding')
    p.add_argument('--pretrained', default='laion2b_s34b_b79k')
    p.add_argument('--only', nargs='*', default=None, help='chi lay video co tien to nay, vd L01 L02')
    return p.parse_args()


def tar_offsets(tar_path):
    """{frame_idx: (offset, size)} cua cac anh .jpg trong tar."""
    out = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith('.jpg'):
                out[int(os.path.basename(m.name)[:-4])] = (m.offset_data, m.size)
    return out


def main():
    args = parse_args()
    data_dirs = [os.path.abspath(d) for d in args.data]
    multi = len(data_dirs) > 1
    out_dir = args.out or os.path.join(data_dirs[0], 'index')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    feats, rows, skipped = [], [], []
    vids, seen = [], {}
    for d in data_dirs:
        map_dir = os.path.join(d, 'map-keyframes')
        if not os.path.isdir(map_dir):
            raise SystemExit(f'Khong thay map-keyframes trong {d}')
        for f in sorted(os.listdir(map_dir)):
            if not f.endswith('.csv'):
                continue
            v = f[:-4]
            if args.only and not any(v.startswith(p) for p in args.only):
                continue
            if v in seen:
                skipped.append((v, f'trung ten, da lay tu {seen[v]}'))
                continue
            seen[v] = d
            vids.append((v, d))
    vids.sort()

    for i, (vid, d) in enumerate(vids, 1):
        tar = os.path.join(d, 'keyframes', f'{vid}.tar')
        npy = os.path.join(d, args.feat_dir, f'{vid}.npy')
        if not (os.path.exists(tar) and os.path.exists(npy)):
            skipped.append((vid, 'thieu tar/npy'))
            continue
        with open(os.path.join(d, 'map-keyframes', f'{vid}.csv'), newline='') as f:
            kf = list(csv.DictReader(f))
        emb = np.load(npy)
        offs = tar_offsets(tar)
        if len(kf) != len(emb) or len(kf) != len(offs):
            skipped.append((vid, f'lech so luong: csv={len(kf)} npy={len(emb)} tar={len(offs)}'))
            continue
        # Gop nhieu thu muc -> luu duong dan tuyet doi. engine.py lam os.path.join(data_dir, tar),
        # ma join voi duong dan tuyet doi se tra ve chinh no, nen engine khong phai sua gi.
        tar_ref = tar.replace(os.sep, '/') if multi else os.path.relpath(tar, d).replace(os.sep, '/')
        for r in kf:
            fi = int(r['frame_idx'])
            off, size = offs[fi]
            rows.append((vid, int(r['n']), fi, float(r['pts_time']), float(r['fps']),
                         tar_ref, off, size))
        feats.append(emb.astype(np.float16))
        if i % 50 == 0 or i == len(vids):
            print(f'  {i}/{len(vids)} video, {len(rows):,} keyframe')

    if not rows:
        raise SystemExit('Khong co keyframe nao -- kiem tra --data')

    # Sap theo (video, frame_idx): keyframe cung video nam lien nhau -> truy van thoi gian chi can searchsorted
    order = sorted(range(len(rows)), key=lambda j: (rows[j][0], rows[j][2]))
    features = np.concatenate(feats)[order]
    rows = [rows[j] for j in order]

    np.save(os.path.join(out_dir, 'features.npy'), features)
    with open(os.path.join(out_dir, 'meta.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['id', 'video', 'n', 'frame_idx', 'pts_time', 'fps', 'tar', 'offset', 'size'])
        for j, r in enumerate(rows):
            wr.writerow([j, *r])
    info = {
        'model': args.model, 'pretrained': args.pretrained, 'dim': int(features.shape[1]),
        'n_keyframes': len(rows), 'n_videos': len({r[0] for r in rows}),
        'data_dir': data_dirs[0], 'data_dirs': data_dirs, 'feat_dir': args.feat_dir,
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out_dir, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f'Xong trong {time.time() - t0:.1f}s: {info["n_keyframes"]:,} keyframe, {info["n_videos"]} video, '
          f'dim {info["dim"]} -> {out_dir}')
    for vid, why in skipped:
        print(f'  !! bo qua {vid}: {why}')


if __name__ == '__main__':
    main()
