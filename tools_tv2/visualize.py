#!/usr/bin/env python3
"""[TV2] Vẽ nhãn YOLO-seg lên ảnh để KIỂM TRA BẰNG MẮT trước khi đưa vào train.

  python visualize.py --data data/roboflow_v1_11cls --out viz_check --n 60
  python visualize.py --data draft_v2 --review draft_v2/review_priority.csv --n 50   # xem ảnh nghi ngờ trước
  python visualize.py --data out_cp --classes 3,6,7 --n 40                           # chỉ ảnh có lớp hiếm

Mỗi ảnh xuất ra gồm 2 nửa: trái = ảnh gốc, phải = ảnh + mask tô màu + tên lớp. Cuối cùng in số lượng instance từng lớp.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv2lib.core import CLASSES, COLORS, find_label, imread, imwrite, list_images, parse_int_list, read_label_file  # noqa: E402

SHORT = ['Car', 'Bus', 'Truck', 'Other', 'Moto', 'MotoRider', 'Bike', 'Cyclist', 'Ped', 'Sign', 'Light']


def draw(img, label_path, alpha=0.45):
    H, W = img.shape[:2]
    over = img.copy()
    out = img.copy()
    items = read_label_file(label_path)
    for c, pts, kind in items:
        if not 0 <= c < len(CLASSES):
            continue
        q = np.round(pts * [W, H] * 16).astype(np.int32)
        cv2.fillPoly(over, [q], COLORS[c], lineType=cv2.LINE_AA, shift=4)
    out = cv2.addWeighted(over, alpha, out, 1 - alpha, 0)
    fs = max(0.35, min(W, H) / 1400)
    for c, pts, kind in items:
        if not 0 <= c < len(CLASSES):
            continue
        q = np.round(pts * [W, H] * 16).astype(np.int32)
        cv2.polylines(out, [q], True, COLORS[c], 1, lineType=cv2.LINE_AA, shift=4)
        x, y = (pts * [W, H]).min(0)
        txt = SHORT[c] + ('[box]' if kind == 'box' else '')
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y = max(int(y), th + 2)
        cv2.rectangle(out, (int(x), y - th - 2), (int(x) + tw + 2, y), COLORS[c], -1)
        cv2.putText(out, txt, (int(x) + 1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return out, items


def legend(width):
    h = 22
    bar = np.full((h, width, 3), 30, np.uint8)
    x = 4
    for i, n in enumerate(SHORT):
        cv2.rectangle(bar, (x, 4), (x + 12, 16), COLORS[i], -1)
        cv2.putText(bar, f'{i}:{n}', (x + 15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        x += 20 + 8 * len(f'{i}:{n}')
        if x > width - 60:
            break
    return bar


def draw_pair_grid(pairs, out_dir, max_side=1280):
    """pairs: list (ảnh, nhãn) -> mỗi ảnh 1 file trái/phải trong out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, (ip, lp) in enumerate(pairs):
        img = imread(ip)
        if img is None:
            continue
        vis, _ = draw(img, lp)
        both = np.hstack([img, vis])
        s = max_side / max(both.shape[:2])
        if s < 1:
            both = cv2.resize(both, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        both = np.vstack([both, legend(both.shape[1])])
        imwrite(out_dir / f'{k:03d}_{Path(ip).stem[:60]}.jpg', both, 90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='Thư mục dataset (có images/ và labels/)')
    ap.add_argument('--out', default=None, help='Thư mục ảnh kiểm tra (mặc định <data>/viz_check)')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--classes', default=None, help="Chỉ lấy ảnh có các lớp này, vd '3,6,7'")
    ap.add_argument('--review', default=None, help='review_priority.csv để xem ảnh ưu tiên cao trước')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    root = Path(a.data)
    imgs = list_images(root)
    want = set(parse_int_list(a.classes)) if a.classes else None
    cnt = [0] * len(CLASSES)
    nimg = [0] * len(CLASSES)
    pool = []
    for p in imgs:
        lp = find_label(p, root)
        items = read_label_file(lp)
        cs = {c for c, _, _ in items if 0 <= c < len(CLASSES)}
        for c, _, _ in items:
            if 0 <= c < len(CLASSES):
                cnt[c] += 1
        for c in cs:
            nimg[c] += 1
        if want is None or cs & want:
            pool.append((p, lp))
    if a.review and Path(a.review).is_file():
        order = [r['image'] for r in csv.DictReader(open(a.review, encoding='utf-8-sig'))]
        by = {str(p.relative_to(root)).replace('\\', '/'): (p, lp) for p, lp in pool}
        pool = [by[o] for o in order if o in by]
    else:
        random.Random(a.seed).shuffle(pool)
    out = Path(a.out) if a.out else root / 'viz_check'
    draw_pair_grid(pool[:a.n], out)
    print(f'{len(imgs)} ảnh. Đã vẽ {min(a.n, len(pool))} ảnh vào {out}')
    print(f'{"id":>2} {"lớp":<14}{"instance":>9}{"số ảnh":>8}')
    for i, c in enumerate(CLASSES):
        warn = '  <-- ÍT' if cnt[i] < 100 else ''
        print(f'{i:>2} {c:<14}{cnt[i]:>9}{nimg[i]:>8}{warn}')


if __name__ == '__main__':
    main()
