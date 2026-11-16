# -*- coding: utf-8 -*-
"""Cat 10 doan video ngau nhien lam de bai GT moi.

Dat ten clip01..clip10 (giau dap an) de khi test lai khong biet truoc la video nao.
Moi clip phai qua kiem tra: tim bang chinh no thi top-1 roi dung video nguon voi diem cao
-> loai nhung doan toan canh studio/MC vi chung khop voi hang tram ban tin khac.

Dau ra <out>/:
    clipNN.mp4        doan 30s, de dua cho NotebookLM
    clipNN_sheet.jpg  9 khung hinh cua clip (phong khi cong cu khong nhan video)
    answers.csv       DAP AN -- khong dua cho NotebookLM
"""
import csv
import os
import random
import cv2

sys.path.insert(0, r'D:\4_Code\IR\search_baseline')
from engine import SearchEngine
from video import sample_frames

INDEX = 'D:/4_Code/IR/Data/keyframes/index'
VIDEO_DIRS = [r'D:\4_Code\IR\Data\batch1', r'D:\4_Code\IR\Data\batch2']
OUT = r'D:\4_Code\IR\search_baseline\runs\gt_clips'
N_CLIPS, CLIP_S, SEED = 10, 30.0, 20261117
MIN_SCORE, MIN_SAME = 0.90, 4                        # nguong coi la doan "dac trung"


def find_video(vid):
    for d in VIDEO_DIRS:
        p = os.path.join(d, vid + '.mp4')
        if os.path.exists(p):
            return p
    return None


def cut(src, t0, dur, dst, max_w=854):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    if w > max_w:
        h = int(h * max_w / w)
        w = max_w
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    n = 0
    while n < int(dur * fps):
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(cv2.resize(fr, (w, h)))
        n += 1
    vw.release()
    cap.release()
    return n / fps if fps else 0


def sheet(path, dst, cols=3, rows=3):
    fr = sample_frames(open(path, 'rb').read(), n=cols * rows)
    W, H = 426, 240
    from PIL import Image
    sh = Image.new('RGB', (W * cols, H * rows), (10, 10, 10))
    for i, im in enumerate(fr[:cols * rows]):
        sh.paste(im.resize((W, H)), ((i % cols) * W, (i // cols) * H))
    sh.save(dst, quality=85)


def main():
    os.makedirs(OUT, exist_ok=True)
    eng = SearchEngine(INDEX)
    m = eng.meta
    rng = random.Random(SEED)
    vids = sorted(m.video.unique())
    rng.shuffle(vids)

    rows, n_done, tried = [], 0, 0
    for vid in vids:
        if n_done >= N_CLIPS:
            break
        src = find_video(vid)
        if not src:
            continue
        sub = m[m.video == vid]
        dur = float(sub.pts_time.max())
        if dur < CLIP_S * 3:
            continue
        for _ in range(3):                           # thu toi da 3 vi tri trong 1 video
            tried += 1
            t0 = rng.uniform(CLIP_S, dur - CLIP_S * 2)
            tmp = os.path.join(OUT, '_tmp.mp4')
            got = cut(src, t0, CLIP_S, tmp)
            if got < CLIP_S * 0.8:
                continue
            res = eng.search_frames(sample_frames(open(tmp, 'rb').read()), 20)
            top = res[0]
            same = sum(1 for r in res[:10] if r['video'] == vid)
            if not (top['video'] == vid and top['score'] >= MIN_SCORE and same >= MIN_SAME):
                continue                             # doan chung chung / canh studio -> bo
            seg = [r for r in res if r['video'] == vid
                   and t0 - 2 <= r['pts_time'] <= t0 + CLIP_S + 2]
            n_done += 1
            name = f'clip{n_done:02d}'
            os.replace(tmp, os.path.join(OUT, name + '.mp4'))
            sheet(os.path.join(OUT, name + '.mp4'), os.path.join(OUT, name + '_sheet.jpg'))
            rows.append({'clip': name, 'video': vid, 't_start_s': round(t0, 1),
                         't_end_s': round(t0 + CLIP_S, 1),
                         'start': min(r['frame_idx'] for r in seg),
                         'end': max(r['frame_idx'] for r in seg),
                         'self_score': round(top['score'], 4), 'same_in_top10': same})
            print(f"  {name}  {vid}  t={t0:.0f}-{t0 + CLIP_S:.0f}s  "
                  f"score={top['score']:.3f}  same={same}/10")
            break

    for junk in (os.path.join(OUT, '_tmp.mp4'),):
        if os.path.exists(junk):
            os.remove(junk)
    with open(os.path.join(OUT, 'answers.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n{len(rows)}/{N_CLIPS} clip ({tried} lan thu) -> {OUT}')
    print('  answers.csv = DAP AN, dung dua cho NotebookLM')


if __name__ == '__main__':
    main()
