"""Web UI + API cho baseline truy van.

    python app.py --index D:/data/HCMAI24_keyframes/index --videos D:/4_Code/IR/batch1
    -> mo http://localhost:8000

--videos (tuy chon): thu muc chua .mp4 (tim de quy) de xem lai doan video quanh keyframe.
--gt: file jsonl luu cac cap (cau truy van, dap an) bam "Luu lam GT" tren UI -> dung cho evaluate.py.
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

import dres
from engine import SearchEngine
from video import sample_frames      # import truc tiep: route 'def video(...)' se che mat module

STATIC = Path(__file__).parent / 'static'


class GTItem(BaseModel):
    query: str
    query_b: str = ''
    query_c: str = ''
    video: str
    frame_idx: int
    pts_time: float
    note: str = ''


class SubmitItem(BaseModel):
    kf_id: int
    text: str = ''      # khac rong -> nop dang VQA


def create_app(index_dir, data_dir=None, video_dir=None, gt_file='queries/gt_ui.jsonl', dres_client=None):
    eng = SearchEngine(index_dir, data_dir)
    videos = {p.stem: p for p in Path(video_dir).rglob('*.mp4')} if video_dir else {}
    app = FastAPI(title='HCMAI24 search baseline')

    def _check(kf_id):
        if not 0 <= kf_id < len(eng.meta):
            raise HTTPException(404, f'id {kf_id} khong ton tai')

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/api/info')
    def info():
        return {**eng.info, 'device': eng.device, 'n_videos_playable': len(videos), 'gt_file': gt_file,
                'dres': dres_client is not None}

    @app.get('/api/search')
    def search(q: str, k: int = 100, max_per_video: int = 0):
        t = time.time()
        res = eng.search(q, k, max_per_video or None)
        return {'results': res, 'ms': round((time.time() - t) * 1000, 1)}

    @app.get('/api/temporal')
    def temporal(a: str, b: str = '', c: str = '', window: float = 30.0,
                 k: int = 100, max_per_video: int = 0):
        t = time.time()
        res = eng.temporal([a, b, c], window, k, max_per_video=max_per_video or None)
        return {'results': res, 'ms': round((time.time() - t) * 1000, 1)}

    @app.post('/api/search_image')
    async def search_image(request: Request, k: int = 100, max_per_video: int = 0, frames_n: int = 0):
        """Body la byte anh HOAC video tho.

        vkis: dan anh chup clip muc tieu, hoac tha thang file 00.mp4 ma capture.py luu ve.
        Video -> lay khung cach deu (frames_n=0: tu chon theo do dai clip),
        diem = MAX tren cac khung roi xep hang nhu truy van anh don.
        """
        raw = await request.body()
        if not raw:
            raise HTTPException(400, 'Body rong, can byte anh hoac video')
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
            frames = [img.convert('RGB')]
        except Exception as img_err:
            try:
                frames = sample_frames(raw, frames_n)         # vkis: ca clip de bai
            except Exception as vid_err:
                raise HTTPException(400, f'Khong doc duoc anh ({img_err}) cung khong phai video ({vid_err})')
        t = time.time()
        res = eng.search_frames(frames, k, max_per_video or None)
        return {'results': res, 'ms': round((time.time() - t) * 1000, 1), 'frames': len(frames)}

    @app.get('/api/similar/{kf_id}')
    def similar(kf_id: int, k: int = 100, max_per_video: int = 0):
        _check(kf_id)
        t = time.time()
        res = eng.similar(kf_id, k, max_per_video or None)
        return {'results': res, 'ms': round((time.time() - t) * 1000, 1)}

    @app.get('/api/neighbors/{kf_id}')
    def neighbors(kf_id: int, r: int = 6):
        _check(kf_id)
        return {'results': eng.neighbors(kf_id, r)}

    @app.get('/img/{kf_id}')
    def img(kf_id: int):
        _check(kf_id)
        return Response(eng.image_bytes(kf_id), media_type='image/jpeg',
                        headers={'Cache-Control': 'public, max-age=86400'})

    @app.get('/video/{vid}')
    def video(vid: str):
        if vid not in videos:
            raise HTTPException(404, 'khong co video (chay app voi --videos)')
        return FileResponse(videos[vid], media_type='video/mp4')

    @app.post('/api/gt')
    def save_gt(item: GTItem):
        os.makedirs(os.path.dirname(gt_file) or '.', exist_ok=True)
        rec = {'id': f'ui_{int(time.time() * 1000)}', 'query': item.query, 'video': item.video,
               'start': item.frame_idx, 'end': item.frame_idx, 'pts_time': item.pts_time,
               'note': item.note, 'created': time.strftime('%Y-%m-%d %H:%M:%S')}
        for f in ('query_b', 'query_c'):
            if getattr(item, f):
                rec[f] = getattr(item, f)
        with open(gt_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return {'ok': True, 'saved_to': os.path.abspath(gt_file)}

    @app.post('/api/submit')
    def submit_dres(item: SubmitItem):
        if dres_client is None:
            raise HTTPException(400, 'Chua cau hinh DRES: chay app voi --dres-url/--dres-user/--dres-pass '
                                     'hoac dat bien moi truong DRES_URL/DRES_USER/DRES_PASS')
        _check(item.kf_id)
        row = eng.meta.iloc[item.kf_id]
        try:
            return dres_client.submit(row['video'], int(row['frame_idx']), float(row['fps']),
                                      text=item.text or None)
        except Exception as e:
            raise HTTPException(502, f'{type(e).__name__}: {e}')

    return app


if __name__ == '__main__':
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True, help='thu muc index (tao boi build_index.py)')
    p.add_argument('--data', default=None, help='OUT_DIR chua keyframes/*.tar (mac dinh lay tu info.json)')
    p.add_argument('--videos', default=None, help='thu muc .mp4 de xem lai video (tuy chon)')
    p.add_argument('--gt', default='queries/gt_ui.jsonl', help='file luu GT khi bam "Luu lam GT"')
    p.add_argument('--host', default='127.0.0.1', help='0.0.0.0 de may khac trong mang truy cap')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--dres-url', default=None, help='vd https://dres.ban-to-chuc.vn (hoac bien DRES_URL)')
    p.add_argument('--dres-user', default=None, help='(hoac bien DRES_USER)')
    p.add_argument('--dres-pass', default=None, help='(hoac bien DRES_PASS) -- nen dung bien moi truong')
    a = p.parse_args()
    client = dres.from_env(a.dres_url, a.dres_user, a.dres_pass)
    print(f'DRES: {client.base if client else "chua cau hinh -> nut Nop bi tat"}')
    uvicorn.run(create_app(a.index, a.data, a.videos, a.gt, client), host=a.host, port=a.port)
