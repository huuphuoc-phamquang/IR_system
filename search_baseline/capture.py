"""Hoi DRES moi 5s, task moi -> luu text + video cua cau hoi vao runs/tasks/<ten task>/.

Chi doc (GET), khong nop gi. Dung chung login/session voi dres.py (DRES_URL, DRES_USER, DRES_PASS).

2 endpoint nay KHONG co trong oas-client.json (lay tu API day du ma trang viewer dung):
    GET /api/v2/evaluation/{evaluationId}/state              -> taskTemplateId, taskStatus
    GET /api/v2/evaluation/{evaluationId}/template/task/{taskTemplateId}/hint -> sequence[{contentType, content, offset}]

Chay:          python capture.py [--interval 5] [--out runs/tasks]
Tu kiem tra:   python capture.py --selftest
"""
import argparse
import base64
import json
import os
import re
import time

from dres import from_env

EXT = {'VIDEO': '.mp4', 'IMAGE': '.jpg'}


def decode_media(content):
    """content la base64, co the kem tien to 'data:video/mp4;base64,'."""
    return base64.b64decode(content.split(',', 1)[1] if content.startswith('data:') else content)


def save_hint(hint, folder):
    """Ghi tung phan tu cua hint ra file; tra ve text gop lai."""
    os.makedirs(folder, exist_ok=True)
    texts = []
    for i, el in enumerate(hint.get('sequence', [])):
        kind, content = str(el.get('contentType')).upper(), el.get('content')
        if not content:
            continue
        if kind == 'TEXT':
            texts.append(content)
        elif kind in EXT:
            with open(os.path.join(folder, f'{i:02d}{EXT[kind]}'), 'wb') as f:
                f.write(decode_media(content))
    text = '\n'.join(texts)
    with open(os.path.join(folder, 'query.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    with open(os.path.join(folder, 'hint.json'), 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in hint.items() if k != 'sequence'} | {'elements': [
            {k: v for k, v in el.items() if k != 'content' or str(el.get('contentType')).upper() == 'TEXT'}
            for el in hint.get('sequence', [])]}, f, ensure_ascii=False, indent=2)
    return text


def get_json(d, path):
    """GET va parse JSON; loi thi bao ro path + HTTP + dau body (DRES tra index.html 200 cho route la)."""
    r = d._req('GET', path)
    try:
        data = r.json()
    except ValueError:
        data = None
    if not r.ok or data is None:
        raise RuntimeError(f'GET {path} -> HTTP {r.status_code}, '
                           f'{r.headers.get("content-type")}: {r.text[:150]!r}')
    return data


def fetch_hint(d, eid, state):
    # cung kieu route voi .../template/task/{templateId}/target/0/media ma viewer goi
    tid = state.get('taskTemplateId')
    if not tid:
        raise RuntimeError(f'State khong co taskTemplateId: {state}')
    hint = get_json(d, f'/api/v2/evaluation/{eid}/template/task/{tid}/hint')
    for el in hint.get('sequence', []):
        if str(el.get('contentType')).upper() not in EXT:
            continue
        # VIDEO/IMAGE: server tra base64 trong 'content', HOAC chi tra link trong 'contentUrl'
        c = el.get('content') or el.get('contentUrl') or ''
        if '/api/' in c:
            r = d._req('GET', c[c.index('/api/'):])
            if not r.ok:
                raise RuntimeError(f'Tai media that bai: GET {c} -> HTTP {r.status_code}')
            el['content'] = base64.b64encode(r.content).decode()
    return hint


def fetch_targets(d, eid, tid, folder, max_targets=5):
    """Dap an viewer chieu SAU khi task ket thuc -> target_<i>.mp4. Chi goi khi task da het RUNNING."""
    n = 0
    for i in range(max_targets):
        r = d._req('GET', f'/api/v2/evaluation/{eid}/template/task/{tid}/target/{i}/media')
        if not r.ok or 'html' in r.headers.get('content-type', ''):
            break
        with open(os.path.join(folder, f'target_{i}.mp4'), 'wb') as f:
            f.write(r.content)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--interval', type=float, default=5)
    p.add_argument('--out', default='runs/tasks')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        return selftest()
    d = from_env()
    if not d:
        raise SystemExit('Thieu DRES_URL / DRES_USER / DRES_PASS')
    seen = None
    pending = None  # (eid, templateId, folder) cua task da lay hint, cho het RUNNING de lay dap an
    while True:
        try:
            evs = get_json(d, '/api/v2/client/evaluation/list')
            ev = next((e for e in evs if str(e.get('status')).upper() == 'ACTIVE'), None)
            if not ev:
                raise RuntimeError('Khong co evaluation nao dang ACTIVE')
            eid = ev['id']
            st = get_json(d, f'/api/v2/evaluation/{eid}/state')
            key = st.get('taskId') or st.get('taskTemplateId')
            running = str(st.get('taskStatus')).upper() == 'RUNNING'
            if pending and (key != seen or not running):
                n = fetch_targets(d, *pending)
                print(f'[{time.strftime("%H:%M:%S")}] dap an: {n} video -> {pending[2]}')
                pending = None
            if key and key != seen and running:
                hint = fetch_hint(d, eid, st)
                name = re.sub(r'[^\w.-]+', '_', st.get('taskTemplateName') or key)
                folder = os.path.join(a.out, name)
                text = save_hint(hint, folder)
                seen, pending = key, (eid, st['taskTemplateId'], folder)
                print(f'[{time.strftime("%H:%M:%S")}] task moi {name} -> {folder}\n  {text}')
        except Exception as e:  # mat mang / chua co evaluation: in ra roi hoi tiep
            print(f'[{time.strftime("%H:%M:%S")}] {e}')
        time.sleep(a.interval)


def selftest():
    import tempfile
    assert decode_media(base64.b64encode(b'abc').decode()) == b'abc'
    assert decode_media('data:video/mp4;base64,' + base64.b64encode(b'xyz').decode()) == b'xyz'
    with tempfile.TemporaryDirectory() as tmp:
        hint = {'taskId': 't1', 'sequence': [
            {'contentType': 'TEXT', 'content': 'a red car', 'offset': 0},
            {'contentType': 'VIDEO', 'content': base64.b64encode(b'MP4').decode(), 'offset': 0},
            {'contentType': 'EMPTY', 'content': None, 'offset': 5}]}
        assert save_hint(hint, tmp) == 'a red car'
        assert open(os.path.join(tmp, '01.mp4'), 'rb').read() == b'MP4'
        assert 'content' not in json.load(open(os.path.join(tmp, 'hint.json')))['elements'][1]

    # vkis that: content=None, link nam o contentUrl -> van phai tai duoc video
    class FakeResp:
        ok, status_code, content = True, 200, b'REAL-MP4'
        def json(self):
            return {'taskId': 't2', 'sequence': [
                {'contentType': 'VIDEO', 'content': None, 'offset': 0,
                 'contentUrl': '/api/v2/evaluation/E/template/task/T/hint/0/media'},
                {'contentType': 'VIDEO', 'content': None, 'offset': 300, 'contentUrl': None}]}

    class FakeDres:
        def __init__(self):
            self.got = []
        def _req(self, method, path, **kw):
            self.got.append(path)
            return FakeResp()

    fd = FakeDres()
    h = fetch_hint(fd, 'E', {'taskTemplateId': 'T'})
    assert fd.got[-1] == '/api/v2/evaluation/E/template/task/T/hint/0/media', fd.got
    assert base64.b64decode(h['sequence'][0]['content']) == b'REAL-MP4'
    assert h['sequence'][1]['content'] is None          # phan tu khong co link thi bo qua, khong no
    with tempfile.TemporaryDirectory() as tmp:
        assert save_hint(h, tmp) == ''
        assert open(os.path.join(tmp, '00.mp4'), 'rb').read() == b'REAL-MP4'

    # dap an: tai target/0, target/1 cho toi khi server bao loi
    class TResp:
        def __init__(self, ok):
            self.ok, self.content, self.headers = ok, b'T', {'content-type': 'video/mp4'}

    class TDres:
        def _req(self, method, path, **kw):
            return TResp(path.endswith(('/target/0/media', '/target/1/media')))

    with tempfile.TemporaryDirectory() as tmp:
        assert fetch_targets(TDres(), 'E', 'T', tmp) == 2
        assert sorted(os.listdir(tmp)) == ['target_0.mp4', 'target_1.mp4']
    print('capture.py selftest OK')


if __name__ == '__main__':
    main()
