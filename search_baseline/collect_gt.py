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

from engine import SearchEngine
from video import sample_frames


def task_rows(tasks_dirs, eng, min_score):
    """Doc tung thu muc task -> dict mo ta, da giai dap an neu co target."""
    if isinstance(tasks_dirs, str):
        tasks_dirs = [tasks_dirs]
    dirs = sorted({d for t in tasks_dirs for d in glob.glob(os.path.join(t, '*'))},
                  key=os.path.basename)
    for d in dirs:
        if not os.path.isdir(d):
            continue
        task = os.path.basename(d)
        qp = os.path.join(d, 'query.txt')
        text = open(qp, encoding='utf-8').read().strip() if os.path.exists(qp) else ''
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        kind = 'vkis' if not lines else ('qakis' if 'áp án' in text else 'tkis')
        targets = sorted(glob.glob(os.path.join(d, 'target_*.mp4')))
        row = {'task': task, 'kind': kind, 'n_hint': len(lines), 'has_target': bool(targets),
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
            near = sorted((r for r in res
                           if r['video'] == top['video'] and r['score'] >= top['score'] - 0.08),
                          key=lambda r: r['pts_time'])
            # chi giu doan LIEN MACH quanh top1: cung video + diem cao chua du, keyframe o doan
            # khac cua cung ban tin se keo span len hang tram giay -> 'khung dung' thanh vo nghia
            i0 = next(i for i, r in enumerate(near) if r['id'] == top['id'])
            lo = hi = i0
            while lo > 0 and near[lo]['pts_time'] - near[lo - 1]['pts_time'] <= 5.0:
                lo -= 1
            while hi + 1 < len(near) and near[hi + 1]['pts_time'] - near[hi]['pts_time'] <= 5.0:
                hi += 1
            block = near[lo:hi + 1]
            row['start'] = min(r['frame_idx'] for r in block)
            row['end'] = max(r['frame_idx'] for r in block)
            row['span_s'] = round(block[-1]['pts_time'] - block[0]['pts_time'], 1)
            row.update(score=round(top['score'], 4), video=top['video'], frame_idx=top['frame_idx'],
                       pts_time=top['pts_time'], kf_id=top['id'],
                       same_in_top10=sum(1 for r in res[:10] if r['video'] == top['video']),
                       margin=round(top['score'] - other['score'], 4) if other else None,
                       solved=top['score'] >= min_score)
        yield row


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
        if r['task'][:8] in skip or r['task'] in skip:
            r['solved'] = False
            r['note'] = 'bo qua theo skip.txt'

    gt = []
    for r in rows:
        if not (r['solved'] and r['lines']):      # vkis khong co chu -> khong lam GT van ban duoc
            continue
        item = {'id': r['task'][:8], 'query': r['lines'][0], 'video': r['video'],
                'start': r['start'], 'end': r['end'], 'span_s': r['span_s'],
                'query_vi': ' | '.join(r['lines']), 'kind': r['kind'],
                'score': r['score'], 'source': 'practice'}
        if len(r['lines']) > 1:
            item['query_b'] = r['lines'][1]
            item['window'] = a.window
        if len(r['lines']) > 2:
            item['query_c'] = r['lines'][2]
        gt.append(item)
        img = os.path.join(a.out, f"{r['task'][:8]}_{r['video']}_{r['frame_idx']}.jpg")
        with open(img, 'wb') as f:
            f.write(eng.image_bytes(r['kf_id']))

    with open(os.path.join(a.out, 'gt.jsonl'), 'w', encoding='utf-8') as f:
        f.write('// Sinh boi collect_gt.py tu vong thi thu. query = cau tieng Anh? KHONG -- van la\n'
                '// tieng Viet, phai dich truoc khi do, hoac bat o "Dich Viet->Anh" tren UI.\n')
        for it in gt:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    cols = ['task', 'kind', 'n_hint', 'has_target', 'score', 'margin', 'same_in_top10', 'solved',
            'video', 'frame_idx', 'pts_time', 'start', 'end', 'span_s', 'note']
    with open(os.path.join(a.out, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(rows, key=lambda r: -(r['score'] or 0)):
            w.writerow(r)

    kinds = {}
    for r in rows:
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1
    solved = [r for r in rows if r['solved']]
    print(f'{len(rows)} task ({kinds}) tu {a.tasks} -> {a.out}')
    print(f'  co clip dap an : {sum(1 for r in rows if r["has_target"])}')
    print(f'  giai duoc (>={a.min_score}): {len(solved)}')
    print(f'  thanh cau GT van ban: {len(gt)}  -> gt.jsonl')
    print(f'  anh kiem tra   : {len(gt)} file .jpg')
    print(f'  bo qua (bo khac / skip.txt): {sum(1 for r in rows if r["has_target"] and not r["solved"])}')
    if skip:
        print(f'  skip.txt loai: {sorted(skip)}')
    print('  >> LIEC QUA CAC FILE .jpg; sai thi them id vao', skip_file)


if __name__ == '__main__':
    main()
