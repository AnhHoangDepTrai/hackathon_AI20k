#!/usr/bin/env python3
"""[TV2] Kiểm tra rò rỉ: ảnh trong tập train (hoặc ảnh sinh ra) trùng / gần trùng golden_val (perceptual hash).

  python check_leak.py --train out_cp --golden golden_val/images
  python check_leak.py --train out_cp --golden golden_val/images --quarantine leak_quarantine   # DI CHUYỂN ảnh rò rỉ ra ngoài

Ảnh bị coi là rò rỉ nếu khoảng cách Hamming pHash <= --thr (mặc định 4; 0 = giống hệt).
Băm trên vùng nội dung (bỏ viền đen letterbox) nên ảnh Roboflow 640x640 vẫn so được với ảnh gốc.
Ảnh tile (cắt 1/4) KHÔNG bắt được bằng hash toàn ảnh -> luôn tile/copy-paste SAU KHI đã loại ảnh golden_val ra.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv2lib.core import HashIndex, find_label, imread, list_images, phash  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', required=True)
    ap.add_argument('--golden', required=True)
    ap.add_argument('--thr', type=int, default=4)
    ap.add_argument('--quarantine', default=None, help='Nếu đặt: di chuyển ảnh + nhãn rò rỉ vào thư mục này')
    a = ap.parse_args()
    gold = HashIndex.from_dir(a.golden)
    root = Path(a.train)
    rows = []
    imgs = list_images(root)
    for p in imgs:
        img = imread(p)
        if img is None:
            continue
        d, name = gold.nearest(phash(img))
        if d <= a.thr:
            rows.append((str(p), name, d))
    out_csv = root / 'leak_report.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['train_image', 'golden_image', 'hamming'])
        w.writerows(rows)
    print(f'Đã kiểm {len(imgs)} ảnh: {len(rows)} ảnh trùng/gần trùng golden_val (thr={a.thr}). Báo cáo: {out_csv}')
    if a.quarantine and rows:
        q = Path(a.quarantine)
        for tp, _, _ in rows:
            tp = Path(tp)
            lp = find_label(tp, root)
            dst = q / tp.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tp), str(dst))
            if lp:
                dl = q / lp.relative_to(root)
                dl.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(lp), str(dl))
        print(f'Đã chuyển {len(rows)} ảnh rò rỉ sang {q}')
    sys.exit(1 if rows and not a.quarantine else 0)


if __name__ == '__main__':
    main()
