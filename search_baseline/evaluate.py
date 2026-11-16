"""Danh gia he thong truy van tren bo cau hoi co dap an.

File cau hoi (.jsonl), moi dong 1 cau:
    {"id": "q1", "query": "a man in a red shirt ...", "video": "L01_V001", "start": 1200, "end": 1500}
    {"id": "q2", "query": "...", "query_b": "...", "window": 30, "video": "L01_V003", "start": 900, "end": 900}
  - start/end: khoang frame_idx dung (bao gom 2 dau). Nut "Luu lam GT" tren UI ghi dung dinh dang nay.
  - query_b (tuy chon): truy van thoi gian A -> B.

Mot ket qua la DUNG neu cung video va frame_idx nam trong [start - tol, end + tol] (tol = --tol giay).

    python evaluate.py --index <data>/index --queries queries/gt_ui.jsonl --out runs/baseline.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from engine import SearchEngine

KS = (1, 5, 10, 20, 50, 100)


def load_queries(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith('//')]


def first_hit(results, q, tol_frames):
    """(rank khung dung dau tien, rank video dung dau tien), 1-based; None neu khong co."""
    frame_rank = video_rank = None
    for rank, r in enumerate(results, 1):
        if r['video'] != q['video']:
            continue
        video_rank = video_rank or rank
        if q['start'] - tol_frames <= r['frame_idx'] <= q['end'] + tol_frames:
            frame_rank = rank
            break
    return frame_rank, video_rank


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', required=True)
    p.add_argument('--data', default=None)
    p.add_argument('--queries', required=True)
    p.add_argument('--k', type=int, default=100, help='so ket qua lay ra moi cau')
    p.add_argument('--tol', type=float, default=2.0, help='dung sai (giay) quanh khoang dap an')
    p.add_argument('--max-per-video', type=int, default=0)
    p.add_argument('--out', default=None, help='CSV ket qua tung cau (de so sanh giua cac lan cai tien)')
    a = p.parse_args()

    eng = SearchEngine(a.index, a.data)
    fps_of = eng.meta.groupby('video')['fps'].first().to_dict()
    qs = load_queries(a.queries)
    rows = []
    for q in qs:
        if q['video'] not in fps_of:
            print(f'  !! bo qua {q.get("id")}: video {q["video"]} khong co trong index')
            continue
        if q.get('query_b'):
            res = eng.temporal([q['query'], q.get('query_b'), q.get('query_c')],
                               q.get('window', 30.0), a.k, max_per_video=a.max_per_video or None)
        else:
            res = eng.search(q['query'], a.k, a.max_per_video or None)
        fr, vr = first_hit(res, q, a.tol * fps_of[q['video']])
        rows.append({'id': q.get('id'), 'query': q['query'], 'video': q['video'],
                     'rank': fr, 'video_rank': vr,
                     'top1': f'{res[0]["video"]}@{res[0]["frame_idx"]}' if res else ''})

    df = pd.DataFrame(rows)
    n = len(df)
    if not n:
        raise SystemExit('Khong co cau hoi hop le')
    rk = df['rank'].astype('float')
    vrk = df['video_rank'].astype('float')
    print(f'\n{n} cau hoi | model {eng.info["model"]} ({eng.info["pretrained"]}) | '
          f'{eng.info["n_keyframes"]:,} keyframe | tol {a.tol}s')
    print('           ' + ''.join(f'R@{k:<6d}' for k in KS) + 'MRR')
    print('khung dung ' + ''.join(f'{(rk <= k).mean():<8.3f}' for k in KS) + f'{np.nan_to_num(1 / rk).mean():.3f}')
    print('video dung ' + ''.join(f'{(vrk <= k).mean():<8.3f}' for k in KS) + f'{np.nan_to_num(1 / vrk).mean():.3f}')
    miss = df[df['rank'].isna()]
    if len(miss):
        print(f'\n{len(miss)} cau khong tim thay trong top {a.k}:')
        for _, r in miss.head(15).iterrows():
            print(f'  {r["id"]}: {r["query"][:80]}  (dap an {r["video"]}, video dung o rank {r["video_rank"]})')
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        df.to_csv(a.out, index=False)
        print('\n->', a.out)


if __name__ == '__main__':
    main()
