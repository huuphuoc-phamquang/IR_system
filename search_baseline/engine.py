"""Loi truy van: text -> CLIP text embedding -> cosine voi toan bo keyframe (brute force).

Vai tram nghin vector x 512 chieu chi la 1 phep nhan ma tran (~10-50 ms ca tren CPU),
nen baseline khong can Milvus/FAISS. Khi so keyframe len hang trieu moi can ANN index.
"""
import json
import os
import threading
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def chain_scores(scores, tkey, cand, window_s):
    """Chuoi su kien dung thu tu: su kien sau nam trong window_s giay sau su kien truoc.

    scores: list mang diem, moi su kien 1 mang. tkey: khoa tang dan (video, thoi gian).
    Tra ve (tong diem cho moi ung vien, {i: [chi so cac su kien sau]}).

    shortcut: tham lam -- moi buoc chon khung diem cao nhat trong cua so roi di tiep, khong
    quay lui. Nang cap sang quy hoach dong neu thay no bo sot chuoi dung.
    """
    total = np.full(len(scores[0]), -np.inf, dtype=np.float64)
    picks = {}
    for i in cand:
        i = int(i)
        s_sum, cur, got = float(scores[0][i]), i, []
        for s in scores[1:]:
            hi = int(np.searchsorted(tkey, tkey[cur] + window_s, side='right'))
            if hi <= cur + 1:
                break                      # het khung trong cua so (hoac sang video khac)
            j = cur + 1 + int(np.argmax(s[cur + 1:hi]))
            s_sum += float(s[j])
            got.append(j)
            cur = j
        total[i] = s_sum                   # chuoi dut giua chung -> it so hang hon -> diem thap hon
        if got:
            picks[i] = got
    return total, picks


class SearchEngine:
    def __init__(self, index_dir, data_dir=None, device=None):
        with open(os.path.join(index_dir, 'info.json'), encoding='utf-8') as f:
            self.info = json.load(f)
        # Uu tien --data truyen vao; neu khong thi lay luc build, cuoi cung la thu muc cha cua index
        cands = [data_dir, self.info.get('data_dir'), os.path.dirname(os.path.abspath(index_dir))]
        self.data_dir = next(d for d in cands if d and os.path.isdir(os.path.join(d, 'keyframes')))

        self.meta = pd.read_csv(os.path.join(index_dir, 'meta.csv'))
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        feats = torch.from_numpy(np.load(os.path.join(index_dir, 'features.npy')))
        self.feats = feats.to(self.device, torch.float16 if self.device == 'cuda' else torch.float32)

        self.video = self.meta['video'].to_numpy()
        self.frame_idx = self.meta['frame_idx'].to_numpy()
        self.pts = self.meta['pts_time'].to_numpy()
        codes, self.video_names = pd.factorize(self.meta['video'], sort=True)
        # Khoa tang dan theo (video, thoi gian) -- index da sap san theo thu tu nay
        self.tkey = codes.astype(np.float64) * 1e6 + self.pts
        assert np.all(np.diff(self.tkey) >= 0), 'meta.csv chua sap theo (video, frame_idx) -- build lai index'

        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.info['model'], pretrained=self.info['pretrained'], device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.info['model'])
        self._lock = threading.Lock()   # model khong an toan khi goi tu nhieu thread cung luc

    # ------------------------------------------------------------------ encode
    @lru_cache(maxsize=2048)
    def _text_vec(self, text):
        with self._lock, torch.no_grad():
            e = self.model.encode_text(self.tokenizer([text]).to(self.device))
        return F.normalize(e.float(), dim=-1)[0]

    def text_scores(self, text):
        q = self._text_vec(text.strip()).to(self.feats.dtype)
        return (self.feats @ q).float().cpu().numpy()

    def id_scores(self, kf_id):
        return (self.feats @ self.feats[kf_id]).float().cpu().numpy()

    # ------------------------------------------------------------------ ranking
    def _rank(self, scores, k, max_per_video=None, extra=None):
        if not max_per_video:
            k = min(k, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
        else:
            # Lay du rong roi loc, tranh 1 video chiem het top-k
            pool = np.argsort(-scores)[:max(k * 20, 2000)]
            count, top = {}, []
            for i in pool:
                v = self.video[i]
                if count.get(v, 0) < max_per_video:
                    count[v] = count.get(v, 0) + 1
                    top.append(i)
                    if len(top) == k:
                        break
            top = np.array(top, dtype=np.int64)
        top = top[np.isfinite(scores[top])]
        return [self.row(int(i), float(scores[i]), **(extra(int(i)) if extra else {})) for i in top]

    def row(self, i, score=None, **kw):
        i = int(i)
        r = {'id': i, 'video': str(self.video[i]), 'frame_idx': int(self.frame_idx[i]),
             'pts_time': round(float(self.pts[i]), 2)}
        if score is not None:
            r['score'] = round(score, 4)
        r.update(kw)
        return r

    def search(self, text, k=100, max_per_video=None):
        return self._rank(self.text_scores(text), k, max_per_video)

    def similar(self, kf_id, k=100, max_per_video=None):
        return self._rank(self.id_scores(kf_id), k, max_per_video)

    def image_scores(self, imgs, chunk=32):
        """Diem cua moi keyframe voi 1 hoac nhieu anh ngoai, lay MAX tren cac anh.

        Dung chinh image tower cua model da tao index nen vector cung khong gian voi features.npy.
        Encode theo lo: 100 khung trong 1 lan goi model nhanh hon han 100 lan goi roi le.
        Cat lo de khong giu ma tran [so_keyframe x so_khung] qua to trong bo nho.
        """
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        best = None
        with self._lock, torch.no_grad():
            for s in range(0, len(imgs), chunk):
                batch = torch.stack([self.preprocess(im) for im in imgs[s:s + chunk]]).to(self.device)
                e = F.normalize(self.model.encode_image(batch).float(), dim=-1).to(self.feats.dtype)
                sc = (self.feats @ e.T).float().amax(dim=1).cpu().numpy()
                best = sc if best is None else np.maximum(best, sc)
        return best

    def search_image(self, img, k=100, max_per_video=None):
        return self._rank(self.image_scores(img), k, max_per_video)

    def search_frames(self, imgs, k=100, max_per_video=None):
        """Nhieu khung cua cung 1 clip. Diem = MAX tren cac khung, khong phai trung binh:
        clip de bai thuong nhieu canh, chi can keyframe khop 1 canh la dung."""
        return self._rank(self.image_scores(imgs), k, max_per_video)

    def temporal(self, texts, window_s=30.0, k=100, n_cand=3000, max_per_video=None):
        """Chuoi su kien theo thu tu (tkis thuong co 3 cau: mo dau / sau do / cuoi cung).

        score = tong sim cua tung su kien; moi su kien sau phai nam trong window_s giay
        sau su kien truoc, cung video. Tra ve keyframe dau chuoi, kem 'next' (su kien 2)
        va 'chain' (toan bo cac su kien sau).
        """
        texts = [t for t in (texts if isinstance(texts, (list, tuple)) else [texts]) if t and t.strip()]
        if not texts:
            raise ValueError('Khong co cau truy van nao')
        scores = [self.text_scores(t) for t in texts]
        if len(scores) == 1:
            return self._rank(scores[0], k, max_per_video)
        cand = np.argpartition(-scores[0], min(n_cand, len(scores[0])) - 1)[:n_cand]
        combined, picks = chain_scores(scores, self.tkey, cand, window_s)

        def extra(i):
            js = picks.get(i)
            if not js:
                return {}
            return {'next': self.row(js[0], float(scores[1][js[0]])),
                    'chain': [self.row(j, float(scores[n + 1][j])) for n, j in enumerate(js)]}

        return self._rank(combined, k, max_per_video, extra=extra)

    def neighbors(self, kf_id, radius=6):
        """Cac keyframe lien truoc/sau trong cung video."""
        lo, hi = max(0, kf_id - radius), min(len(self.video), kf_id + radius + 1)
        return [self.row(i) for i in range(lo, hi) if self.video[i] == self.video[kf_id]]

    # ------------------------------------------------------------------ anh
    def image_bytes(self, kf_id):
        r = self.meta.iloc[kf_id]
        with open(os.path.join(self.data_dir, r['tar']), 'rb') as f:
            f.seek(int(r['offset']))
            return f.read(int(r['size']))


if __name__ == '__main__':
    # 2 video, moi video 6 keyframe cach nhau 10s -> tkey = ma_video*1e6 + giay
    tkey = np.array([0, 10, 20, 30, 40, 50, 1e6, 1e6 + 10, 1e6 + 20, 1e6 + 30, 1e6 + 40, 1e6 + 50])
    n = len(tkey)

    def arr(**kw):
        a = np.zeros(n)
        for i, v in kw.items():
            a[int(i)] = v
        return a

    # A@0, B@20, C@40 trong video 1 -> chuoi dung, cua so 25s du de nhay tung buoc
    A, B, C = arr(**{'0': 1.0, '6': 0.9}), arr(**{'2': 1.0}), arr(**{'4': 1.0})
    tot, picks = chain_scores([A, B, C], tkey, [0, 6], 25.0)
    assert picks[0] == [2, 4], picks
    assert abs(tot[0] - 3.0) < 1e-9, tot[0]
    # ung vien 6 (video 2) khong co B/C -> chi con diem A, phai thua ung vien 0
    assert abs(tot[6] - 0.9) < 1e-9 and tot[0] > tot[6]

    # cua so qua hep -> khong bat duoc B
    tot2, picks2 = chain_scores([A, B, C], tkey, [0], 5.0)
    assert 0 not in picks2 and abs(tot2[0] - 1.0) < 1e-9

    # khong duoc nhay sang video khac: A@50 (video 1), B chi co o video 2
    A2, B2 = arr(**{'5': 1.0}), arr(**{'6': 1.0})
    tot3, picks3 = chain_scores([A2, B2], tkey, [5], 100.0)
    assert 5 not in picks3, picks3

    # phai dung THU TU: C nam TRUOC B thi khong duoc tinh diem
    C_truoc = arr(**{'1': 1.0})                        # C dinh o t=10, truoc B@20
    tot4, picks4 = chain_scores([A, B, C_truoc], tkey, [0], 25.0)
    assert picks4[0][0] == 2, picks4                   # B van bat dung @20
    assert picks4[0][1] > 2, picks4                    # C chi duoc tim SAU B
    assert abs(tot4[0] - 2.0) < 1e-9, tot4[0]          # 1.0(A) + 1.0(B) + 0(C bi loai)

    # 1 su kien -> giong het tim thuong
    tot5, picks5 = chain_scores([A], tkey, [0, 6], 25.0)
    assert picks5 == {} and tot5[0] == 1.0 and tot5[6] == 0.9
    print('engine.py chain_scores selftest OK')
