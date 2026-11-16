"""Gom cac task da bat duoc o vong thi thu thanh 1 thu muc, kem dap an da giai duoc.

Moi thu muc runs/tasks/<task>/ co the co:
    query.txt      3 cau mo ta (tkis/qakis), rong neu la vkis
    target_*.mp4   clip DAP AN, DRES chieu sau khi task ket thuc
Dung chinh engine de tim clip dap an trong index -> ra (video, frame_idx). Diem >= --min-score
thi coi la giai duoc; thap hon nghia la dap an thuoc bo du lieu khac, bo qua.

Dau ra (--out, mac dinh runs/gt/):
    gt.jsonl       cac cau tkis/qakis co dap an -> dung thang cho evaluate.py
    summary.csv    MOI task kem diem va ket luan, ke ca cai bi loai
    <task>.jpg     keyframe dap an, de liec mat kiem tra nhanh

PHAI LIEC QUA CAC ANH DO. Diem cao khong dam bao dung: clip dap an hay mo dau bang canh
studio/MC dung chung cho moi ban tin, khop 0.96 voi hang tram ban tin khac. Da do thu: margin
(top1 tru video khac tot nhat) KHONG tach duoc ca sai khoi ca dung, nen khong loc tu dong duoc.
Thay anh nao sai -> them task id vao <out>/skip.txt (moi dong 1 id), chay lai se bo qua.

    python collect_gt.py --index D:/4_Code/IR/Data/keyframes/index
"""
import argparse
import csv
import glob
import json
import os
import shutil

import cv2

from engine import SearchEngine
from video import sample_frames


def clip_seconds(path):
    """Do dai clip dap an (giay); 0 neu khong doc duoc."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return n / fps if fps else 0.0
    finally:
        cap.release()


def task_rows(tasks_dirs, eng, min_score):
    """Doc tung thu muc task -> dict mo ta, da giai dap an neu co target."""
    if isinstance(tasks_dirs, str):
        tasks_dirs = [tasks_dirs]
    dirs = sorted({d for t in tasks_dirs for d in glob.glob(os.path.join(t, '*'))},
                  key=os.path.basename)
    # DRES chay lai cung mot de -> taskId moi moi lan -> capture.py tao them thu muc.
    # 'taskId' trong hint.json la id cua TEMPLATE, on dinh -> gom theo no, giu ban tot nhat
    # (uu tien co clip dap an, roi nhieu cau hint hon, roi moi nhat).
    groups = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        hj = os.path.join(d, 'hint.json')
        try:
            tid = json.load(open(hj, encoding='utf-8')).get('taskId')
        except Exception:
            tid = None
        key = tid or os.path.basename(d)
        qp = os.path.join(d, 'query.txt')
        n_lines = len([l for l in open(qp, encoding='utf-8').read().splitlines() if l.strip()])             if os.path.exists(qp) else 0
        rank = (bool(glob.glob(os.path.join(d, 'target_*.mp4'))), n_lines, os.path.getmtime(d))
        if key not in groups or rank > groups[key][0]:
            groups[key] = (rank, d)
    tid_of = {d: k for k, (_, d) in groups.items()}
    for _, d in sorted(groups.values(), key=lambda x: os.path.basename(x[1])):
        task = os.path.basename(d)
        qp = os.path.join(d, 'query.txt')
        text = open(qp, encoding='utf-8').read().strip() if os.path.exists(qp) else ''
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        kind = 'vkis' if not lines else ('qakis' if 'áp án' in text else 'tkis')
        targets = sorted(glob.glob(os.path.join(d, 'target_*.mp4')))
        row = {'task': task, 'tid': tid_of.get(d), 'kind': kind, 'n_hint': len(lines), 'has_target': bool(targets),
               'score': None, 'video': None, 'frame_idx': None, 'pts_time': None,
               'same_in_top10': None, 'solved': False, 'lines': lines}
        if targets:
            try:
                res = eng.search_frames(sample_frames(open(targets[0], 'rb').read()), 50)
            except Exception as e:
                row['note'] = f'{type(e).__name__}: {e}'
                yield row
                continue
            top = res[0]
            other = next((r for r in res if r['video'] != top['video']), None)
            # Doan dap an dai 10-25s, khong phai 1 khung. Lay khoang frame_idx cua cac keyframe
            # cung video co diem gan top -> GT la mot DOAN, danh gia moi co y nghia.
            # Dap an la CA DOAN clip, khong phai 1 khung. Khung khop nhat voi clip thuong roi
            # vao canh noi bat nhat (cau B hoac C), trong khi truy van chuoi tra ve khung cua
            # cau A -> cach nhau hang chuc giay. Lay ca doan dai bang clip quanh khung do.
            dur = clip_seconds(targets[0]) or 20.0
            t0, t1 = top['pts_time'] - dur, top['pts_time'] + dur
            seg = [r for r in res if r['video'] == top['video'] and t0 <= r['pts_time'] <= t1]
            row['start'] = min(r['frame_idx'] for r in seg)
            row['end'] = max(r['frame_idx'] for r in seg)
            row['span_s'] = round(max(r['pts_time'] for r in seg) - min(r['pts_time'] for r in seg), 1)
            row['clip_s'] = round(dur, 1)
            row.update(score=round(top['score'], 4), video=top['video'], frame_idx=top['frame_idx'],
                       pts_time=top['pts_time'], kf_id=top['id'],
                       same_in_top10=sum(1 for r in res[:10] if r['video'] == top['video']),
                       margin=round(top['score'] - other['score'], 4) if other else None,
                       solved=top['score'] >= min_score)
        yield row


def write_xlsx(path, gt, img_of):
    """Mot dong moi cau, kem anh keyframe nhung trong o -> soat bang mat trong 1 file."""
    try:
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        print('  (bo qua .xlsx: thieu openpyxl -- pip install openpyxl)')
        return
    from PIL import Image as PILImage

    wb = Workbook()
    ws = wb.active
    ws.title = 'GT'
    head = ['STT', 'id', 'SAI?', 'keyframe', 'video', 'start', 'end', 'span_s', 'score',
            'Cau 1 (VI)', 'Cau 2 (VI)', 'Cau 3 (VI)', 'Cau 1 (EN)', 'Cau 2 (EN)', 'Cau 3 (EN)',
            'thu muc task']
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill('solid', fgColor='DDDDDD')
    widths = [5, 11, 7, 34, 11, 8, 8, 8, 8, 46, 46, 46, 46, 46, 46, 38]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i) if i <= 26 else 'A' + chr(38 + i)].width = w

    tmp = os.path.join(os.path.dirname(path), '_thumbs')
    os.makedirs(tmp, exist_ok=True)
    for n, it in enumerate(gt, 1):
        vi = (it.get('query_vi') or '').split(' | ')
        en = it.get('en') or []
        ws.append([n, it['id'], '', '', it['video'], it['start'], it['end'],
                   it.get('span_s'), it.get('score'),
                   *(vi + ['', '', ''])[:3], *(list(en) + ['', '', ''])[:3],
                   it.get('task_dir', '')])
        r = ws.max_row
        ws.row_dimensions[r].height = 102          # ~136 px cho anh cao 135
        for col in ('J', 'K', 'L', 'M', 'N', 'O'):
            ws[f'{col}{r}'].alignment = Alignment(wrap_text=True, vertical='top')
        src = img_of.get(it['id'])
        if src and os.path.exists(src):
            th = os.path.join(tmp, f"{it['id']}.png")
            im = PILImage.open(src).convert('RGB')
            im.thumbnail((240, 135))
            im.save(th)
            ws.add_image(XLImage(th), f'D{r}')
    ws.freeze_panes = 'C2'
    wb.save(path)
    shutil.rmtree(tmp, ignore_errors=True)   # anh da nam trong .xlsx, khong can file tam
    print(f'  -> {path}  ({len(gt)} dong, cot "SAI?" de ban danh dau)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', required=True)
    p.add_argument('--tasks', nargs='+', default=['runs/tasks', 'runs/tasks_done'],
                   help='quet nhieu thu muc; mac dinh ca task moi lan task da luu kho')
    p.add_argument('--out', default='runs/gt')
    p.add_argument('--min-score', type=float, default=0.90,
                   help='diem toi thieu de tin dap an nam trong index (do duoc: trung >=0.93, truot <=0.85)')
    p.add_argument('--window', type=float, default=60.0, help='window cho cau hoi chuoi su kien')
    a = p.parse_args()

    eng = SearchEngine(a.index)
    os.makedirs(a.out, exist_ok=True)
    skip_file = os.path.join(a.out, 'skip.txt')
    skip = {l.split('#')[0].strip() for l in open(skip_file, encoding='utf-8')} if os.path.exists(skip_file) else set()
    skip.discard('')
    rows = list(task_rows(a.tasks, eng, a.min_score))
    for r in rows:
        ids = {r['task'], r['task'][:8], r.get('tid'), (r.get('tid') or '')[:8]}
        if ids & skip:
            r['solved'] = False
            r['note'] = 'bo qua theo skip.txt'

    gt = []
    for r in rows:
        if not (r['solved'] and r['lines']):      # vkis khong co chu -> khong lam GT van ban duoc
            continue
        # id theo taskId cua template, on dinh qua cac lan chay lai; ten thu muc thi khong
        item = {'id': (r.get('tid') or r['task'])[:8], 'task_dir': r['task'],
                'query': r['lines'][0], 'video': r['video'],
                'start': r['start'], 'end': r['end'], 'span_s': r['span_s'], 'clip_s': r.get('clip_s'),
                'query_vi': ' | '.join(r['lines']), 'kind': r['kind'],
                'score': r['score'], 'source': 'practice'}
        if len(r['lines']) > 1:
            item['query_b'] = r['lines'][1]
            item['window'] = a.window
        if len(r['lines']) > 2:
            item['query_c'] = r['lines'][2]
        gt.append(item)
        img = os.path.join(a.out, f"{item['id']}_{r['video']}_{r['frame_idx']}.jpg")
        with open(img, 'wb') as f:
            f.write(eng.image_bytes(r['kf_id']))

    # vkis khong co chu -> GT rieng, truy van bang chinh clip de bai (00.mp4).
    # Khong vong vo: dap an dung tu target_*.mp4, con truy van la 00.mp4 -- hai file khac nhau.
    vkis = []
    for r in rows:
        if not r['solved'] or r['lines']:
            continue
        cand = [c for t in a.tasks for c in glob.glob(os.path.join(t, r['task'], '00.mp4'))]
        if not cand:
            continue
        vkis.append({'id': (r.get('tid') or r['task'])[:8], 'clip': cand[0].replace(os.sep, '/'),
                     'video': r['video'], 'start': r['start'], 'end': r['end'],
                     'span_s': r['span_s'], 'score': r['score'], 'kind': 'vkis',
                     'task_dir': r['task'], 'source': 'practice'})
    with open(os.path.join(a.out, 'gt_vkis.jsonl'), 'w', encoding='utf-8') as f:
        f.write('// vkis: truy van bang clip de bai 00.mp4, dap an giai tu target_*.mp4' + chr(10))
        for it in vkis:
            f.write(json.dumps(it, ensure_ascii=False) + chr(10))

    with open(os.path.join(a.out, 'gt.jsonl'), 'w', encoding='utf-8') as f:
        f.write('// Sinh boi collect_gt.py tu vong thi thu. query = cau tieng Anh? KHONG -- van la\n'
                '// tieng Viet, phai dich truoc khi do, hoac bat o "Dich Viet->Anh" tren UI.\n')
        for it in gt:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    cols = ['task', 'kind', 'n_hint', 'has_target', 'score', 'margin', 'same_in_top10', 'solved',
            'video', 'frame_idx', 'pts_time', 'start', 'end', 'span_s', 'clip_s', 'tid', 'note']
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(rows, key=lambda r: -(r['score'] or 0)):
            w.writerow(r)

    # ghep ban dich tieng Anh neu da co, de Excel xem duoc ca hai thu tieng
    en_path = os.path.join(a.out, 'gt_en.jsonl')
    en_of = {}
    if os.path.exists(en_path):
        for line in open(en_path, encoding='utf-8'):
            if line.startswith('//'):
                continue
            j = json.loads(line)
            en_of[j['id']] = [j.get(k, '') for k in ('query', 'query_b', 'query_c')]
    img_of = {}
    for it in gt:
        it['en'] = en_of.get(it['id'], [])
        hit = glob.glob(os.path.join(a.out, f"{it['id']}_*.jpg"))
        if hit:
            img_of[it['id']] = hit[0]
    write_xlsx(os.path.join(a.out, 'gt.xlsx'), gt, img_of)

    kinds = {}
    for r in rows:
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1
    solved = [r for r in rows if r['solved']]
    print(f'{len(rows)} task ({kinds}) tu {a.tasks} -> {a.out}')
    print(f'  co clip dap an : {sum(1 for r in rows if r["has_target"])}')
    print(f'  giai duoc (>={a.min_score}): {len(solved)}')
    print(f'  thanh cau GT van ban: {len(gt)}  -> gt.jsonl')
    print(f'  thanh cau GT bang clip: {len(vkis)}  -> gt_vkis.jsonl')
    print(f'  anh kiem tra   : {len(gt)} file .jpg')
    print(f'  bo qua (bo khac / skip.txt): {sum(1 for r in rows if r["has_target"] and not r["solved"])}')
    if skip:
        print(f'  skip.txt loai: {sorted(skip)}')
    print('  >> LIEC QUA CAC FILE .jpg; sai thi them id vao', skip_file)


if __name__ == '__main__':
    main()
