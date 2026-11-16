"""Bien mo ta do NotebookLM (hoac nguoi) viet thanh bo GT chay duoc voi evaluate.py.

Dau vao:
    --text     file .txt dan nguyen output, dinh dang:
                   clip01
                   Mo dau la ...
                   Sau do ...
                   Cuoi cung la ...
                   (dong trong)
                   clip02
                   ...
    --answers  runs/gt_clips/answers.csv (clip, video, start, end, ...)

Dau ra: gt_clips.jsonl -- query/query_b/query_c la TIENG VIET, nho bat o "Dich Viet->Anh"
tren UI hoac dich truoc khi do, vi ViT-B-32 khong hieu tieng Viet.

    python descriptions_to_gt.py --text runs/gt_clips/mota.txt
    python evaluate.py --index <index> --queries runs/gt_clips/gt_clips.jsonl --tol 3

Tu kiem tra: python descriptions_to_gt.py --selftest
"""
import argparse
import csv
import json
import os
import re


def parse(text):
    """-> {clip: [cau1, cau2, cau3]}. Bo qua dong trong, dau liet ke, va ``` cua markdown."""
    out, cur = {}, None
    for raw in text.splitlines():
        line = raw.strip().lstrip('-*0123456789. ').strip()
        if not line or line.startswith('```'):
            continue
        m = re.fullmatch(r'(?:\*\*)?(clip\d+)(?:\*\*)?:?', line, re.I)
        if m:
            cur = m.group(1).lower()
            out[cur] = []
            continue
        if re.match(r'(?:\*\*)?clip', line, re.I):
            cur = None          # dong giong tieu de nhung khong doc duoc -> ket thuc clip hien tai
            continue
        if cur is not None and len(out[cur]) < 3:   # chi can 3 cau; chan van rac dinh vao cuoi
            out[cur].append(line)
    return {k: v for k, v in out.items() if v}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--text')
    p.add_argument('--answers', default='runs/gt_clips/answers.csv')
    p.add_argument('--out', default=None, help='mac dinh canh --answers')
    p.add_argument('--window', type=float, default=60.0)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.text:
        raise SystemExit('Thieu --text')

    ans = {r['clip']: r for r in csv.DictReader(open(a.answers, encoding='utf-8'))}
    got = parse(open(a.text, encoding='utf-8').read())
    out = a.out or os.path.join(os.path.dirname(a.answers), 'gt_clips.jsonl')

    rows, skipped = [], []
    for clip, lines in sorted(got.items()):
        if clip not in ans:
            skipped.append((clip, 'khong co trong answers.csv'))
            continue
        if len(lines) < 1:
            skipped.append((clip, 'khong co cau nao'))
            continue
        r = ans[clip]
        item = {'id': clip, 'query': lines[0], 'video': r['video'],
                'start': int(r['start']), 'end': int(r['end']),
                'source': 'notebooklm', 'kind': 'tkis'}
        if len(lines) > 1:
            item['query_b'] = lines[1]
            item['window'] = a.window
        if len(lines) > 2:
            item['query_c'] = lines[2]
        rows.append(item)

    with open(out, 'w', encoding='utf-8') as f:
        f.write('// Sinh boi descriptions_to_gt.py. Cau la TIENG VIET -> phai dich truoc khi do.\n')
        for it in rows:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    print(f'{len(rows)}/{len(ans)} clip -> {out}')
    for c, why in skipped:
        print(f'  !! {c}: {why}')
    missing = sorted(set(ans) - set(got))
    if missing:
        print(f'  !! thieu mo ta cho: {missing}')


def selftest():
    txt = """Day la phan tra loi:

```
**clip01**
Mo dau la hai nguoi dan ong dang khieng mot thung go
Sau do camera lia sang mot chiec xe tai mau trang
Cuoi cung la canh nhieu nguoi dung xem ben duong
```

clip03:
1. Mo dau la mot can phong hop dong nguoi
2. Sau do mot nguoi phu nu ao do dung len phat bieu

clipXX
khong phai clip hop le
"""
    g = parse(txt)
    assert set(g) == {'clip01', 'clip03'}, g
    assert len(g['clip01']) == 3 and g['clip01'][0].startswith('Mo dau la hai nguoi')
    assert len(g['clip03']) == 2, g['clip03']
    assert 'khieng' in g['clip01'][0] and 'xe tai' in g['clip01'][1]
    assert not any(l.startswith('1.') for l in g['clip03'])     # da bo dau liet ke
    print('descriptions_to_gt.py selftest OK')


if __name__ == '__main__':
    main()
