"""Lay vai khung hinh cach deu tu mot video, de dung ca clip lam truy van.

Dung cho task vkis: clip de bai (capture.py luu ra <task>/00.mp4) thay vi 1 anh chup man hinh.

Tu kiem tra (khong can mang, khong can index):  python video.py
"""
import os
import tempfile

import cv2
from PIL import Image


def _to_pil(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def sample_frames(data: bytes, n=0, every_s=0.5, max_n=30, min_n=4):
    """Khung hinh cach deu -> list PIL.Image RGB. Khong doc duoc -> ValueError.

    n=0 (mac dinh): 2 khung/giay, toi da max_n khung -> clip dai hon 15s thi tu giam mat do
    (30 khung trai deu ca clip) de khong no so lan goi model.
    Clip de bai dai tu 0.5s den vai phut nen so khung CO DINH la sai o ca hai dau:
    clip 369s voi 8 khung thi moi 46 giay moi lay 1 -> bo sot gan het canh.
    """
    if n < 0:
        raise ValueError('n phai >= 0')
    fd, path = tempfile.mkstemp(suffix='.mp4')   # cv2.VideoCapture can duong dan file
    os.close(fd)
    try:
        with open(path, 'wb') as f:
            f.write(data)
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise ValueError('cv2 khong mo duoc video')
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if n == 0:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                dur = total / fps if total > 0 else 0
                n = max(min_n, min(max_n, round(dur / every_s))) if dur else min_n
            out = []
            if total > 0:
                # lay giua moi doan -> tranh khung dau/cuoi hay bi den
                for i in range(n):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(total * (i + 0.5) / n), total - 1))
                    ok, fr = cap.read()
                    if ok:
                        out.append(_to_pil(fr))
            if not out:
                # shortcut: container khong bao so frame -> chi quet ~n*15 khung dau.
                # Nang cap khi gap clip dai ma header thieu nb_frames.
                cap.release()
                cap = cv2.VideoCapture(path)
                i = 0
                while len(out) < n:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if i % 15 == 0:
                        out.append(_to_pil(fr))
                    i += 1
            if not out:
                raise ValueError('Video khong co khung hinh nao doc duoc')
            return out
        finally:
            cap.release()
    finally:
        os.unlink(path)


if __name__ == '__main__':
    import numpy as np

    def make_mp4(n_frames, w=160, h=120):
        fd, p = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))
        for i in range(n_frames):
            fr = np.full((h, w, 3), i * 255 // max(n_frames - 1, 1), dtype=np.uint8)
            fr[:, :, 0] = 0          # kenh B = 0 -> kiem tra thu tu kenh BGR->RGB
            vw.write(fr)
        vw.release()
        data = open(p, 'rb').read()
        os.unlink(p)
        return data

    data = make_mp4(60)                       # 60 khung @10fps = 6 giay
    fr = sample_frames(data, 8)
    assert len(fr) == 8, len(fr)
    assert all(f.mode == 'RGB' and f.size == (160, 120) for f in fr)
    # anh sang tang dan theo thoi gian -> cac khung phai khac nhau va tang dan
    means = [np.asarray(f).mean() for f in fr]
    assert means == sorted(means), means
    assert means[-1] - means[0] > 20, means
    # BGR->RGB: kenh B goc = 0 -> sau chuyen doi phai nam o vi tri 2
    a = np.asarray(fr[-1])
    assert a[:, :, 2].mean() < 5 and a[:, :, 0].mean() > 50, (a[:, :, 0].mean(), a[:, :, 2].mean())

    assert len(sample_frames(data, 1)) == 1
    assert len(sample_frames(make_mp4(3), 8)) <= 8          # video ngan hon so khung yeu cau

    # n=0: 2 khung/giay, kep trong [4, 30]
    assert len(sample_frames(make_mp4(60))) == 12           # 6s -> 12 khung (2/giay)
    assert len(sample_frames(make_mp4(150))) == 30          # 15s -> 30 khung, van du 2/giay
    assert len(sample_frames(make_mp4(10))) == 4            # 1s -> 2 khung nhung kep len min 4
    assert len(sample_frames(make_mp4(300))) == 30          # 30s -> kep xuong 30 (1/giay)

    for bad in (b'', b'khong phai video'):
        try:
            sample_frames(bad)
            raise AssertionError('phai bao loi')
        except ValueError:
            pass
    print('video.py selftest OK')
