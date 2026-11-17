"""Encode lai keyframe bang mot model open_clip khac (vd SigLIP2), khong dung toi bo cu.

Doc anh thang tu keyframes/<vid>.tar, ghi <data>/<feat-dir>/<vid>.npy (float16, L2-norm) theo
dung thu tu dong cua map-keyframes/<vid>.csv -> build_index.py dung duoc ngay:

    python encode_features.py --data D:/.../Batch2 --model ViT-L-16-SigLIP2-384 --pretrained webli \
                              --feat-dir features-siglip2-l16-384
    python build_index.py --data D:/.../Batch2 --feat-dir features-siglip2-l16-384 \
                          --model ViT-L-16-SigLIP2-384 --pretrained webli --out D:/.../index_siglip2

Chay lai se bo qua video da co file .npy, nen dut giua chung thi cu chay tiep.
"""
import argparse
import csv
import io
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from build_index import tar_offsets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, nargs='+', help='thu muc chua keyframes/ va map-keyframes/')
    p.add_argument('--model', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--feat-dir', required=True, help='ten thu muc dac trung moi, vd features-siglip2-l16-384')
    p.add_argument('--only', nargs='*', default=None, help='chi video co tien to nay, vd L21 L22')
    p.add_argument('--batch', type=int, default=48)
    a = p.parse_args()

    import open_clip
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained, device=dev)
    model.eval()

    jobs = []
    for d in a.data:
        for f in sorted(os.listdir(os.path.join(d, 'map-keyframes'))):
            vid = f[:-4]
            if f.endswith('.csv') and (not a.only or any(vid.startswith(x) for x in a.only)):
                jobs.append((d, vid))
    todo = [(d, v) for d, v in jobs if not os.path.exists(os.path.join(d, a.feat_dir, v + '.npy'))]
    print(f'{len(jobs)} video, con {len(todo)} chua encode | {a.model} ({a.pretrained}) tren {dev}', flush=True)

    t0, n_img = time.time(), 0
    for k, (d, vid) in enumerate(todo, 1):
        out_dir = os.path.join(d, a.feat_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(d, 'map-keyframes', vid + '.csv'), newline='') as f:
            frames = [int(r['frame_idx']) for r in csv.DictReader(f)]
        tar = os.path.join(d, 'keyframes', vid + '.tar')
        offs = tar_offsets(tar)
        feats = []
        with open(tar, 'rb') as fh, torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
            for s in range(0, len(frames), a.batch):
                imgs = []
                for fi in frames[s:s + a.batch]:        # dung thu tu cua csv
                    off, size = offs[fi]
                    fh.seek(off)
                    imgs.append(preprocess(Image.open(io.BytesIO(fh.read(size))).convert('RGB')))
                e = model.encode_image(torch.stack(imgs).to(dev))
                feats.append(F.normalize(e.float(), dim=-1).half().cpu().numpy())
        emb = np.concatenate(feats)
        assert emb.shape[0] == len(frames), (vid, emb.shape, len(frames))
        tmp = os.path.join(out_dir, vid + '.tmp.npy')
        np.save(tmp, emb)
        os.replace(tmp, os.path.join(out_dir, vid + '.npy'))    # khong de lai file do neu bi ngat
        n_img += len(frames)
        if k % 10 == 0 or k == len(todo):
            rate = n_img / (time.time() - t0)
            print(f'  {k}/{len(todo)} video, {n_img:,} anh, {rate:.0f} anh/s', flush=True)
    print(f'Xong: {n_img:,} anh trong {(time.time() - t0) / 60:.1f} phut', flush=True)


if __name__ == '__main__':
    main()
