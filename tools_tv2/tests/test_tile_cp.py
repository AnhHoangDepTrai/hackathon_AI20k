"""Kiểm thử tiling + copy-paste trên dữ liệu giả lập:  python tests/test_tile_cp.py"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tv2lib.core import Inst, read_label_file  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')


def write(d, name, img, lines):
    (d / 'images').mkdir(parents=True, exist_ok=True)
    (d / 'labels').mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / 'images' / f'{name}.jpg'), img)
    (d / 'labels' / f'{name}.txt').write_text('\n'.join(lines) + '\n')


def poly_line(c, pts, W, H):
    return f'{c} ' + ' '.join(f'{x/W:.6f} {y/H:.6f}' for x, y in pts)


def label_masks(lbl, W, H):
    out = []
    for c, pts, _ in read_label_file(lbl):
        it = Inst.from_polygon(pts * [W, H], H, W, cls=c)
        out.append(it)
    return out


def test_tiling(tmp):
    W, H = 1920, 1080
    img = np.random.default_rng(0).integers(60, 200, (H, W, 3), dtype=np.uint8)
    lines = []
    signs = [(100, 100), (930, 500), (1000, 520), (1700, 900)]  # có biển nằm đúng vùng chồng lấn
    for x, y in signs:
        pts = [(x, y), (x + 22, y), (x + 22, y + 22), (x, y + 22)]
        cv2.rectangle(img, (x, y), (x + 22, y + 22), (0, 0, 255), -1)
        lines.append(poly_line(9, pts, W, H))
    big = [(800, 400), (1150, 400), (1150, 700), (800, 700)]   # ô tô lớn bị cắt bởi biên ô
    lines.append(poly_line(0, big, W, H))
    write(tmp / 'hi', 'frame', img, lines)
    r = subprocess.run([PY, str(ROOT / 'tile_images.py'), '--data', str(tmp / 'hi'), '--out', str(tmp / 'tiles')],
                       capture_output=True, text=True, encoding='utf-8', errors='replace', env=ENV)
    print(r.stdout[-600:], r.stderr[-2000:])
    tiles = sorted((tmp / 'tiles' / 'images').glob('*.jpg'))
    assert len(tiles) == 4, tiles
    total_signs = 0
    for t in tiles:
        ti = cv2.imread(str(t))
        th, tw = ti.shape[:2]
        assert abs(tw - 1067) <= 1 and abs(th - 600) <= 1, (tw, th)
        for it in label_masks(tmp / 'tiles' / 'labels' / (t.stem + '.txt'), tw, th):
            if it.cls == 9:
                total_signs += 1
                patch = ti[it.oy:it.oy + it.mask.shape[0], it.ox:it.ox + it.mask.shape[1]][it.mask]
                assert patch[:, 2].mean() > 200, 'polygon biển báo lệch khỏi vật thật'
    # 2 biển ở vùng chồng lấn xuất hiện ở nhiều ô -> tổng >= 4
    assert total_signs >= 4, total_signs
    print(f'tiling OK: 4 ô 1067x600, {total_signs} instance biển báo, polygon khớp đúng pixel')


def test_copy_paste(tmp):
    rng = np.random.default_rng(1)
    W, H = 1280, 720
    # nguồn: xe đạp (lớp 6) và cyclist (7) hình elip màu
    for k in range(6):
        img = rng.integers(150, 220, (H, W, 3), dtype=np.uint8)
        m = np.zeros((H, W), np.uint8)
        cx, cy = 300 + 100 * k, 450
        cv2.ellipse(m, (cx, cy), (40, 60), 0, 0, 360, 1, -1)
        img[m > 0] = (0, 200, 0)
        it = Inst.from_mask(m.astype(bool), cls=6 + (k % 2))
        p = it.polygon_px()
        write(tmp / 'src', f's{k}', img, [poly_line(it.cls, p, W, H)])
    # nền tối (đêm) có 3 ô tô theo phối cảnh
    for k in range(5):
        img = rng.integers(10, 50, (H, W, 3), dtype=np.uint8)
        lines = []
        for (x, y1, h) in [(200, 500, 80), (600, 600, 140), (950, 690, 200)]:
            cv2.rectangle(img, (x, y1 - h), (x + int(1.6 * h), y1), (90, 90, 90), -1)
            lines.append(poly_line(0, [(x, y1 - h), (x + 1.6 * h, y1 - h), (x + 1.6 * h, y1), (x, y1)], W, H))
        write(tmp / 'bg', f'b{k}', img, lines)
    r = subprocess.run([PY, str(ROOT / 'copy_paste.py'), '--src', str(tmp / 'src'), '--bg', str(tmp / 'bg'),
                        '--out', str(tmp / 'cp'), '--n', '8', '--bg-mode', 'night', '--per-image', '2', '3',
                        '--rain-fx', '0.5', '--classes', '6,7'], capture_output=True, text=True, encoding='utf-8', errors='replace', env=ENV)
    print(r.stdout[-900:], r.stderr[-2000:])
    assert r.returncode == 0
    outs = sorted((tmp / 'cp' / 'images').glob('*.jpg'))
    assert len(outs) == 8, len(outs)
    n_pasted = 0
    for o in outs:
        insts = label_masks(tmp / 'cp' / 'labels' / (o.stem + '.txt'), W, H)
        n_pasted += sum(i.cls in (6, 7) for i in insts)
        full = [i.full(H, W) for i in insts]
        for a in range(len(full)):
            for b in range(a + 1, len(full)):
                ov = (full[a] & full[b]).sum()
                assert ov <= 0.02 * min(full[a].sum(), full[b].sum()) + 30, f'chồng lấn {ov}px ở {o.name}'
    assert n_pasted >= 8
    print(f'copy-paste OK: {len(outs)} ảnh, {n_pasted} vật lớp hiếm, không chồng lấn')


if __name__ == '__main__':
    tmp = Path(tempfile.mkdtemp())
    try:
        test_tiling(tmp)
        test_copy_paste(tmp)
        print('\nTILING + COPY-PASTE ĐẠT')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
