"""Gom moi nguon GT ve mot bo chuan, chia theo loai task.

    runs/gt_std/
        vkis/v01/   query.mp4    clip de bai (DRES chieu)
                    answer.json  video goc + khoang frame/giay
        tkis/t01/   query.txt    3 doan mo ta (tieng Viet)
                    query_en.txt 3 doan da dich (neu co)
                    answer.mp4   doan video dap an
                    answer.json  video goc + khoang frame/giay
        qakis/q01/  query.txt    cau hoi
                    answer.json  dap an (co the chua biet)
        index.csv                bang tong hop ca 3 loai

Chay lai duoc nhieu lan, ghi de. Nguon doc tu runs/gt/*.jsonl va runs/gt_clips/*.jsonl
(sinh boi collect_gt.py / descriptions_to_gt.py).

    python standardize_gt.py --index D:/4_Code/IR/Data/keyframes/index
"""
import argparse
import csv
import glob
import json
import os
import shutil


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding='utf-8') if not l.lstrip().startswith('//')]


def copy_if_changed(src, dst):
    """Bo qua neu dich da ton tai va cung kich thuoc -- chay lai khong chep lai 283 MB."""
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return False
    shutil.copy(src, dst)
    return True


def find_task_file(task_dir, pattern):
    for root in ('runs/tasks', 'runs/tasks_done'):
        hit = sorted(glob.glob(os.path.join(root, task_dir, pattern)))
        if hit:
            return hit[0]
    return None


def seconds_of(sec_map, video, frame):
    return sec_map.get((video, int(frame)))


def merge_kept(new, kept):
    """Giu gia tri cu o nhung truong ban moi bo trong, va moi truong nguoi dung tu them.

    shortcut: gia tri script tinh duoc luon thang gia tri cu. Neu can sua tay de len ket qua
    tu dong thi phai them co chong ghi de.
    """
    if not kept:
        return new
    out = dict(new)
    for k, v in kept.items():
        if out.get(k) is None and v is not None:
            out[k] = v
    return out


def write_answer(folder, item, sec_map, extra=None, kept=None):
    a = {'video': item['video'],
         'start_frame': int(item['start']), 'end_frame': int(item['end']),
         'start_s': seconds_of(sec_map, item['video'], item['start']),
         'end_s': seconds_of(sec_map, item['video'], item['end']),
         'source': item.get('source'), 'match_score': item.get('score')}
    a.update(extra or {})
    a = merge_kept(a, kept)
    with open(os.path.join(folder, 'answer.json'), 'w', encoding='utf-8') as f:
        json.dump(a, f, ensure_ascii=False, indent=2)
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', required=True, help='de tra giay tu frame_idx (meta.csv)')
    p.add_argument('--out', default='runs/gt_std')
    a = p.parse_args()

    import pandas as pd
    meta = pd.read_csv(os.path.join(a.index, 'meta.csv'))
    sec_map = {(v, int(f)): round(float(t), 2)
               for v, f, t in zip(meta.video, meta.frame_idx, meta.pts_time)}

    # Giu lai gia tri nguoi dung tu dien (vd answer_text cua qakis). Khoa theo task_id/task_dir
    # chu khong theo ten thu muc: them cau moi la t01 tro sang cau khac, sua tay se dinh nham.
    kept = {}
    for old_file in glob.glob(os.path.join(a.out, '*', '*', 'answer.json')):
        try:
            j = json.load(open(old_file, encoding='utf-8'))
        except Exception:
            continue
        key = j.get('task_id') or j.get('task_dir')
        if key:
            kept[key] = j
    # Khong rmtree ca cay: tren Windows no co the chet giua chung vi file .mp4 dang bi khoa,
    # de lai bo GT xoa do. Ghi de tung file, cuoi cung moi don thu muc thua.
    rows = []

    # ---- vkis: clip de bai + dap an ----
    for n, it in enumerate(load('runs/gt/gt_vkis.jsonl'), 1):
        d = os.path.join(a.out, 'vkis', f'v{n:02d}')
        os.makedirs(d, exist_ok=True)
        copy_if_changed(it['clip'], os.path.join(d, 'query.mp4'))
        ans = write_answer(d, it, sec_map, {'task_id': it['id'], 'task_dir': it.get('task_dir')},
                           kept.get(it['id']))
        rows.append({'folder': f'vkis/v{n:02d}', 'kind': 'vkis', 'id': it['id'], **ans})

    # ---- tkis: 3 doan mo ta + doan video dap an ----
    tkis = []
    en_of = {j['id']: j for j in load('runs/gt/gt_en.jsonl')}
    for it in load('runs/gt/gt.jsonl'):
        clip = find_task_file(it.get('task_dir', ''), 'target_*.mp4')
        tkis.append((it, en_of.get(it['id']), clip))
    en_clip = {j['id']: j for j in load('runs/gt_clips/gt_clips_en.jsonl')}
    for it in load('runs/gt_clips/gt_clips.jsonl'):
        tkis.append((it, en_clip.get(it['id']), os.path.join('runs/gt_clips', it['id'] + '.mp4')))

    for n, (it, en, clip) in enumerate(tkis, 1):
        d = os.path.join(a.out, 'tkis', f't{n:02d}')
        os.makedirs(d, exist_ok=True)
        vi = [it.get(k) for k in ('query', 'query_b', 'query_c') if it.get(k)]
        open(os.path.join(d, 'query.txt'), 'w', encoding='utf-8').write('\n'.join(vi) + '\n')
        if en:
            lines = [en.get(k) for k in ('query', 'query_b', 'query_c') if en.get(k)]
            open(os.path.join(d, 'query_en.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
        if clip and os.path.exists(clip):
            copy_if_changed(clip, os.path.join(d, 'answer.mp4'))
        ans = write_answer(d, it, sec_map, {'task_id': it['id'], 'n_hint': len(vi),
                                            'has_en': bool(en)}, kept.get(it['id']))
        rows.append({'folder': f'tkis/t{n:02d}', 'kind': 'tkis', 'id': it['id'], **ans})

    # ---- qakis: cau hoi, dap an con de trong ----
    n = 0
    seen_q = set()
    for qp in sorted(glob.glob('runs/tasks*/*/query.txt')):
        txt = open(qp, encoding='utf-8').read().strip()
        if 'áp án' not in txt or txt in seen_q:
            continue
        seen_q.add(txt)
        n += 1
        d = os.path.join(a.out, 'qakis', f'q{n:02d}')
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'query.txt'), 'w', encoding='utf-8').write(txt + '\n')
        task_dir = os.path.basename(os.path.dirname(qp))
        clip = find_task_file(task_dir, 'target_*.mp4')
        if clip:
            copy_if_changed(clip, os.path.join(d, 'answer.mp4'))
        # qakis phai DOC dap an tren man hinh -> khong tu dong giai duoc, de nguoi dien
        ans = {'video': None, 'start_frame': None, 'end_frame': None,
               'answer_text': None, 'task_dir': task_dir,
               'note': 'Dap an la CHU doc tren man hinh. Tim canh nhu tkis, doc dap an, '
                       'dien vao answer_text roi nop dang text tren UI.'}
        ans = merge_kept(ans, kept.get(task_dir))
        json.dump(ans, open(os.path.join(d, 'answer.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        rows.append({'folder': f'qakis/q{n:02d}', 'kind': 'qakis', 'id': task_dir[:8],
                     'video': None, 'start_frame': None, 'end_frame': None})

    made = {r['folder'].replace('/', os.sep) for r in rows}
    for d in sorted(glob.glob(os.path.join(a.out, '*', '*'))):
        if os.path.isdir(d) and os.path.relpath(d, a.out) not in made:
            shutil.rmtree(d, ignore_errors=True)    # thu muc cua lan chay truoc, gio khong con cau nao
            print('  don thu muc thua:', os.path.relpath(d, a.out))

    cols = ['folder', 'kind', 'id', 'video', 'start_frame', 'end_frame', 'start_s', 'end_s',
            'match_score', 'source', 'n_hint', 'has_en', 'task_id', 'task_dir']
    with open(os.path.join(a.out, 'index.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    # kiem tra: moi thu muc phai du file bat buoc
    bad = []
    for r in rows:
        d = os.path.join(a.out, r['folder'])
        need = {'vkis': ['query.mp4', 'answer.json'],
                'tkis': ['query.txt', 'answer.json'],
                'qakis': ['query.txt', 'answer.json']}[r['kind']]
        for fn in need:
            if not os.path.exists(os.path.join(d, fn)):
                bad.append(f"{r['folder']}/{fn}")
    n_clip = sum(1 for r in rows if r['kind'] == 'tkis'
                 and os.path.exists(os.path.join(a.out, r['folder'], 'answer.mp4')))
    k = {x: sum(1 for r in rows if r['kind'] == x) for x in ('vkis', 'tkis', 'qakis')}
    print(f"{len(rows)} cau -> {a.out}   vkis {k['vkis']} | tkis {k['tkis']} | qakis {k['qakis']}")
    print(f"  tkis co answer.mp4: {n_clip}/{k['tkis']}")
    print('  THIEU FILE: ' + ', '.join(bad) if bad else '  du file o moi thu muc')
    if kept:
        print(f'  giu lai gia tri tu dien cua {len(kept)} cau tu lan chay truoc')


if __name__ == '__main__':
    main()
