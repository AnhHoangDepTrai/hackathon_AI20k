#!/usr/bin/env python3
"""[TV2] Tiling: cắt ảnh độ phân giải cao thành lưới 2x2 có chồng lấn (mặc định 20%) để vật nhỏ
(Sign, Traffic light) không bị "biến mất" khi YOLO resize ảnh về 640.

Vì sao: ảnh 1920x1080 resize về 640 -> mọi thứ nhỏ đi 3 lần; biển báo 24px còn 8px (gần như mất).
Cắt 2x2 (mỗi ô ~1067x600) -> khi resize về 640 vật chỉ nhỏ đi ~1.7 lần -> mô hình học được chi tiết.

Polygon được CẮT GỌT đúng theo biên ô (raster -> contour), vật bị cắt còn quá ít (< --min-visible phần
diện tích VÀ nhỏ) thì bỏ, vật quá nhỏ theo ngưỡng cuộc thi thì bỏ.

  python tile_images.py --data data/vn_frames_labeled --out data/tiles_v2
  python tile_images.py --data X --out Y --require-classes all --min-side 1000 --overlap 0.15

Mặc định chỉ tile ảnh có cạnh dài >= 1280 VÀ có ít nhất 1 Sign/Traffic light nhỏ (cạnh ngắn < 40px khi về 640).
Ảnh Roboflow 640x640 của nhóm sẽ bị bỏ qua (đã nhỏ sẵn, tile vô nghĩa).
!!! Chỉ tile ảnh TRAIN. Không bao giờ tile ảnh golden_val (tile của ảnh val lọt vào train = rò rỉ).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv2lib.core import (CLASSES, HashIndex, Inst, find_label, format_line, imread, imwrite, label_rel_for,  # noqa: E402
                         list_images, mask_to_polygon, now, out_img_rel, parse_int_list, phash, progress,
                         read_label_file, save_json, tight, too_small, write_data_yaml)


def tile_grid(W, H, n, overlap):
    """Kích thước ô = W / (n - (n-1)*overlap) => 2 ô kề nhau chồng nhau đúng `overlap` x kích thước ô."""
    tw = int(math.ceil(W / (n - (n - 1) * overlap)))
    th = int(math.ceil(H / (n - (n - 1) * overlap)))
    xs = [round(i * (W - tw) / (n - 1)) if n > 1 else 0 for i in range(n)]
    ys = [round(j * (H - th) / (n - 1)) if n > 1 else 0 for j in range(n)]
    return [(r, c, xs[c], ys[r], tw, th) for r in range(n) for c in range(n)]


def needs_tiling(items, W, H, req, small_px):
    if req is None:
        return True
    s = 640.0 / max(W, H)
    for c, pts, _ in items:
        if c in req:
            px = pts * [W, H]
            wh = px.max(0) - px.min(0)
            if min(wh) * s < small_px:
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='Dataset 11 lớp (images/ + labels/) — CHỈ tập train')
    ap.add_argument('--out', required=True)
    ap.add_argument('--grid', type=int, default=2, help='2 = lưới 2x2')
    ap.add_argument('--overlap', type=float, default=0.20, help='Tỉ lệ chồng lấn giữa 2 ô kề (0.15–0.20)')
    ap.add_argument('--min-side', type=int, default=1280, help='Chỉ tile ảnh có cạnh dài >= giá trị này')
    ap.add_argument('--require-classes', default='9,10', help="Chỉ tile ảnh có lớp này ở dạng nhỏ; 'all' = tile mọi ảnh đủ lớn")
    ap.add_argument('--small-px', type=float, default=40, help='"Nhỏ" = cạnh ngắn < N px sau khi resize ảnh về 640')
    ap.add_argument('--min-visible', type=float, default=0.30, help='Giữ vật bị cắt nếu còn >= tỉ lệ này diện tích...')
    ap.add_argument('--min-keep-area', type=float, default=400, help='...hoặc phần còn lại >= N px (thang 640 của ô)')
    ap.add_argument('--keep-empty', action='store_true', help='Giữ cả ô không có vật nào (ảnh nền)')
    ap.add_argument('--copy-original', action='store_true', help='Chép cả ảnh gốc sang --out (để out dùng độc lập)')
    ap.add_argument('--golden-val', default=None, help='Thư mục ảnh golden_val: bỏ qua ảnh nguồn trùng/gần trùng')
    ap.add_argument('--quality', type=int, default=95)
    a = ap.parse_args()

    root, out = Path(a.data), Path(a.out)
    req = None if a.require_classes.strip().lower() == 'all' else set(parse_int_list(a.require_classes))
    gold = HashIndex.from_dir(a.golden_val)
    imgs = list_images(root)
    t0 = time.time()
    n_src = n_tiles = n_small = n_skip_req = n_leak = 0
    inst_cnt = [0] * len(CLASSES)
    dropped_partial = 0
    for k, p in enumerate(imgs, 1):
        progress(k, len(imgs), t0, every=100, prefix='  ')
        lp = find_label(p, root)
        items = read_label_file(lp)
        img = imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        rel = p.relative_to(root)
        if a.copy_original:
            imwrite(out / out_img_rel(rel), img, a.quality)
            dst = out / label_rel_for(out_img_rel(rel))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(lp.read_text(encoding='utf-8') if lp else '', encoding='utf-8')
        if max(H, W) < a.min_side:
            n_small += 1
            continue
        if not needs_tiling(items, W, H, req, a.small_px):
            n_skip_req += 1
            continue
        if gold.names:
            d, _ = gold.nearest(phash(img))
            if d <= 4:
                n_leak += 1
                continue
        insts = []
        for c, pts, _ in items:
            it = Inst.from_polygon(pts * [W, H], H, W, cls=c)
            if it is not None:
                insts.append(it)
        n_src += 1
        for (r, c, x0, y0, tw, th) in tile_grid(W, H, a.grid, a.overlap):
            tile = img[y0:y0 + th, x0:x0 + tw]
            th_, tw_ = tile.shape[:2]
            s640 = 640.0 / max(th_, tw_)
            lines = []
            for it in insts:
                full = it.area
                m = it.crop(x0, y0, x0 + tw_, y0 + th_)
                m, ox, oy = tight(m, 0, 0)
                if m is None:
                    continue
                part = Inst(cls=it.cls, mask=m, ox=ox, oy=oy)
                frac = part.area / max(1, full)
                if too_small(part, th_, tw_):
                    continue
                if frac < a.min_visible and part.area * s640 ** 2 < a.min_keep_area:
                    dropped_partial += 1
                    continue
                poly = mask_to_polygon(part.mask, part.ox, part.oy)
                if poly is None or len(poly) < 3:
                    continue
                lines.append(format_line(it.cls, poly / [tw_, th_]))
                inst_cnt[it.cls] += 1
            if not lines and not a.keep_empty:
                continue
            o_rel = out_img_rel(rel)
            o_rel = o_rel.with_name(f'{o_rel.stem}_tile{r}{c}{o_rel.suffix}')
            imwrite(out / o_rel, tile, a.quality)
            lf = out / label_rel_for(o_rel)
            lf.parent.mkdir(parents=True, exist_ok=True)
            lf.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
            n_tiles += 1
    out.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out)
    info = dict(tool='tools_tv2/tile_images.py', time=now(), input=str(root), output=str(out), grid=a.grid,
                overlap=a.overlap, min_side=a.min_side, require_classes=a.require_classes, small_px=a.small_px,
                n_images=len(imgs), n_source_tiled=n_src, n_tiles=n_tiles, skipped_too_small_image=n_small,
                skipped_no_small_target=n_skip_req, skipped_golden_leak=n_leak, dropped_partial_instances=dropped_partial,
                instances_in_tiles={CLASSES[i]: inst_cnt[i] for i in range(len(CLASSES))})
    save_json(out / 'tile_log.json', info)
    print(f'\nẢnh: {len(imgs)} | đã tile: {n_src} -> {n_tiles} ô | bỏ (ảnh nhỏ hơn {a.min_side}px): {n_small} | '
          f'bỏ (không có vật nhỏ lớp {a.require_classes}): {n_skip_req} | bỏ (trùng golden_val): {n_leak}')
    print('Instance trong các ô: ' + ', '.join(f'{CLASSES[i]}={inst_cnt[i]}' for i in range(len(CLASSES)) if inst_cnt[i]))
    if n_src == 0:
        print('!!! Không ảnh nào được tile. Ảnh 640x640 (Roboflow) không cần tile. Thử --min-side nhỏ hơn hoặc --require-classes all.')


if __name__ == '__main__':
    main()
