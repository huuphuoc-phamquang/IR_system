"""Nop ket qua len DRES (theo oas-client.json 2.0.0-RC4).

3 call HTTP:
    POST /api/v2/login                        {username, password}  -> sessionId
    GET  /api/v2/client/evaluation/list       ?session=             -> lay cai status == ACTIVE
    POST /api/v2/submit/{evaluationId}        ?session=             -> ket qua cham

Session cache lai; gap 401 thi login lai 1 lan.

Tu kiem tra (khong can mang):  python dres.py
"""
import os

import requests


def frame_to_ms(frame_idx, fps):
    """DRES nhan start/end bang milliseconds.

    fps phai la fps cua chinh video do, khong duoc hard-code 25: batch2 co 187 video
    o 30fps va 1 video o 26.44fps -> lech 20% tuong duong hang tram giay.
    """
    return round(frame_idx / fps * 1000)


def build_body(video, frame_idx, fps, text=None):
    """ApiClientSubmission. text != None -> nop dang VQA thay vi media item."""
    ms = frame_to_ms(frame_idx, fps)
    answer = {'text': text} if text else {'mediaItemName': video, 'start': ms, 'end': ms}
    # taskName/taskId la optional: server tu suy ra task dang chay. Neu bi 412 thi phai
    # goi /api/v2/client/evaluation/currentTask/{id} roi dien taskName vao day.
    return {'answerSets': [{'answers': [answer]}]}, ms


class Dres:
    def __init__(self, base_url, username, password, timeout=15):
        self.base = base_url.rstrip('/')
        self.user, self.pwd, self.timeout = username, password, timeout
        self.session_id = None

    def _login(self):
        r = requests.post(f'{self.base}/api/v2/login',
                          json={'username': self.user, 'password': self.pwd}, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f'Login that bai: HTTP {r.status_code} {r.text[:200]}')
        try:
            self.session_id = r.json().get('sessionId')
        except ValueError:  # thuong do DRES_URL tro vao trang web chu khong phai goc API
            raise RuntimeError(f'Login tra ve khong phai JSON ({r.url}): {r.text[:200]!r}')
        if not self.session_id:
            raise RuntimeError(f'Login khong tra ve sessionId: {r.text[:200]}')
        return self.session_id

    def _req(self, method, path, **kw):
        for attempt in (1, 2):
            kw.setdefault('params', {})['session'] = self.session_id or self._login()
            r = requests.request(method, f'{self.base}{path}', timeout=self.timeout, **kw)
            if r.status_code == 401 and attempt == 1:
                self.session_id = None      # het han -> login lai 1 lan
                continue
            return r

    def active_evaluation(self):
        r = self._req('GET', '/api/v2/client/evaluation/list')
        if not r.ok:
            raise RuntimeError(f'Lay evaluation list that bai: HTTP {r.status_code} {r.text[:200]}')
        ev = next((e for e in r.json() if str(e.get('status')).upper() == 'ACTIVE'), None)
        if not ev:
            raise RuntimeError('Khong co evaluation nao dang ACTIVE')
        return ev['id'], ev.get('name', ev['id'])

    def submit(self, video, frame_idx, fps, text=None):
        eid, name = self.active_evaluation()
        body, ms = build_body(video, frame_idx, fps, text)
        r = self._req('POST', f'/api/v2/submit/{eid}', json=body)
        try:
            data = r.json()
        except ValueError:
            data = {'description': r.text[:300]}
        return {'http': r.status_code, 'ok': r.ok, 'evaluation': name, 'ms': ms,
                'status': data.get('status'), 'submission': data.get('submission'),
                'description': data.get('description')}


def from_env(url=None, user=None, pwd=None):
    """Tra ve Dres neu du 3 thong tin (uu tien tham so, sau do bien moi truong), nguoc lai None."""
    url = url or os.getenv('DRES_URL')
    user = user or os.getenv('DRES_USER')
    pwd = pwd or os.getenv('DRES_PASS')
    return Dres(url, user, pwd) if (url and user and pwd) else None


if __name__ == '__main__':
    assert frame_to_ms(30000, 25.0) == 1200000
    assert frame_to_ms(30000, 30.0) == 1000000, 'video 30fps bi tinh theo 25fps -> lech 200s'
    assert frame_to_ms(933, 26.44) == 35287
    assert frame_to_ms(0, 25.0) == 0
    body, ms = build_body('L01_V003', 11400, 25.0)
    assert body == {'answerSets': [{'answers': [
        {'mediaItemName': 'L01_V003', 'start': 456000, 'end': 456000}]}]}, body
    assert ms == 456000
    body, _ = build_body('L01_V003', 11400, 25.0, text='QA-abc')
    assert body == {'answerSets': [{'answers': [{'text': 'QA-abc'}]}]}, body
    assert from_env(None, None, None) is None or os.getenv('DRES_URL')
    assert from_env('http://x:5000/', 'u', 'p').base == 'http://x:5000'
    print('dres.py selftest OK')
