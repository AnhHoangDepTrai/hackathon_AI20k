#!/usr/bin/env python3
"""[TV2] Copy-Paste augmentation (dữ liệu tổng hợp): cắt instance LỚP HIẾM (mặc định 3 Other vehicle, 6 Bicycle,
7 Cyclist) từ ảnh đã gán nhãn, dán sang ảnh nền khác (ưu tiên nền ĐÊM / MƯA) -> ảnh mới + nhãn mới.

Làm gì để ảnh dán "thật" và nhãn ĐÚNG:
  1. Ngân hàng instance: chỉ lấy vật nguyên vẹn (không chạm mép ảnh), đủ lớn, mask không quá rỗng.
  2. Vị trí: vật mặt đất (xe, người) đặt ở nửa dưới khung hình; vật cao (biển/đèn) ở nửa trên. Bỏ viền đen letterbox.
  3. Tỉ lệ theo phối cảnh: ước lượng "pixel / mét" theo toạ độ y từ chính các xe/người có sẵn trong ảnh nền
     (fallback: mô hình đường chân trời) -> xe đạp ở xa thì nhỏ, ở gần thì to. Không phóng to quá --max-up lần.
  4. Hoà màu (LAB) theo vùng nền xung quanh -> vật dán ban ngày vào nền đêm sẽ tối và ám màu theo nền.
  5. Viền mềm (feather) khi trộn ảnh, nhưng NHÃN dùng mask cứng.
  6. Zero-Overlap: vật dán nằm trước -> trừ phần bị che khỏi mask các vật cũ; không cho che > 50% vật cũ.
  7. Tuỳ chọn: hiệu ứng mưa (--rain-fx) / biến nền ngày thành đêm (--night-fx).

  python copy_paste.py --src data/v1_11cls/train --bg data/v1_11cls/train --out data/cp_v1 --n 400 --bg-mode night
  python copy_paste.py --src data/coco_bike_11cls data/v1_11cls/train --bg data/rain_frames --out data/cp_rain --n 300
  python copy_paste.py ... --classes 3,6,7,10 --per-image 2 4 --rain-fx 0.3 --golden-val golden_val/images

Khuyến nghị: ảnh tổng hợp chiếm <= 20–30% tập train; KHÔNG BAO GIỜ đưa vào val.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv2lib.core import (CLASSES, ELEVATED, GROUND, PRIOR_HEIGHT_M, SIGN, HashIndex, Inst, content_box,  # noqa: E402
                         find_label, format_line, imread, imwrite, list_images, now, parse_int_list, phash,
                         read_label_file, resolve_overlaps, save_json, too_small, write_data_yaml)


# ----------------------------------------------------------------------------------------------- ngân hàng instance
def build_bank(src_dirs, classes, gold, min_h=24, log=print):
    bank = {c: [] for c in classes}
    n_img = n_leak = 0
    for root in src_dirs:
        root = Path(root)
        for p in list_images(root):
            lp = find_label(p, root)
            items = [(c, pts) for c, pts, kind in read_label_file(lp) if c in classes and kind == 'poly']
            if not items:
                continue
            img = imread(p)
            if img is None:
                continue
            if gold.names and gold.nearest(phash(img))[0] <= 4:
                n_leak += 1
                continue
            n_img += 1
            H, W = img.shape[:2]
            cx0, cy0, cx1, cy1 = content_box(img)
            for c, pts in items:
                it = Inst.from_polygon(pts * [W, H], H, W, cls=c)
                if it is None:
                    continue
                x0, y0, x1, y1 = it.box
                h, w = y1 - y0, x1 - x0
                if h < min_h or w < 8:
                    continue
                if x0 <= cx0 + 1 or y0 <= cy0 + 1 or x1 >= cx1 - 1 or y1 >= cy1 - 1:
                    continue  # chạm mép -> vật bị cắt, dán sẽ trông cụt
                fill = it.area / float(h * w)
                if fill < (0.10 if c in (6, 7) else 0.25):
                    continue
                n_comp = cv2.connectedComponents(it.mask.astype(np.uint8))[0] - 1
                if n_comp > 3:
                    continue  # bị che nhiều mảnh
                rgb = img[y0:y1, x0:x1].copy()
                bank[c].append(dict(rgb=rgb, mask=it.mask.copy(), src=str(p), y_bottom_rel=(y1 - cy0) / max(1, cy1 - cy0)))
    log(f'[bank] {n_img} ảnh nguồn, bỏ {n_leak} ảnh trùng golden_val. Số instance: ' +
        ', '.join(f'{c}:{CLASSES[c]}={len(v)}' for c, v in bank.items()))
    return bank


def bank_preview(bank, path, per_class=24, cell=96):
    """Ảnh xem nhanh ngân hàng instance: vật nào mask xấu (lẹm, dính nền) thì loại nguồn đó trước khi dán hàng loạt."""
    rows = []
    for c, items in bank.items():
        if not items:
            continue
        row = np.full((cell + 18, cell * per_class, 3), 40, np.uint8)
        cv2.putText(row, f'{c}:{CLASSES[c]} ({len(items)})', (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for k, it in enumerate(items[:per_class]):
            h, w = it['mask'].shape
            s = (cell - 4) / max(h, w)
            nh, nw = max(1, int(h * s)), max(1, int(w * s))
            rgb = cv2.resize(it['rgb'], (nw, nh))
            m = cv2.resize(it['mask'].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
            tile = np.full((nh, nw, 3), 128, np.uint8)
            tile[m] = rgb[m]
            row[18 + 2:18 + 2 + nh, k * cell + 2:k * cell + 2 + nw] = tile
        rows.append(row)
    if rows:
        imwrite(path, np.vstack(rows), 90)


# ----------------------------------------------------------------------------------------------- phối cảnh
def fit_ppm(bg_insts, cbox):
    """pixel/mét theo y (đáy vật): ppm(y) = a*(y - y_h). Ước lượng từ xe/người có sẵn trong ảnh nền."""
    cx0, cy0, cx1, cy1 = cbox
    ys, ppm = [], []
    for it in bg_insts:
        if it.cls not in PRIOR_HEIGHT_M:
            continue
        x0, y0, x1, y1 = it.box
        if y0 <= cy0 + 2 or y1 >= cy1 - 2 or x0 <= cx0 + 2 or x1 >= cx1 - 2:
            continue
        ys.append(y1)
        ppm.append((y1 - y0) / PRIOR_HEIGHT_M[it.cls])
    ch = cy1 - cy0
    hz_def = cy0 + 0.42 * ch
    a_def = (0.30 * ch / 1.6) / max(1.0, cy1 - hz_def)
    if len(ys) >= 3:
        ys, ppm = np.array(ys, float), np.array(ppm, float)
        A = np.stack([ys, np.ones_like(ys)], 1)
        try:
            (a, b), *_ = np.linalg.lstsq(A, ppm, rcond=None)
            if a > 0:
                hz = -b / a
                if cy0 - 0.2 * ch < hz < cy0 + 0.75 * ch:
                    return float(a), float(hz)
        except np.linalg.LinAlgError:
            pass
        # chỉ ước lượng hệ số độ lớn, giữ đường chân trời mặc định
        a = float(np.median(ppm / np.maximum(ys - hz_def, 1.0)))
        if a > 0:
            return a, hz_def
    return a_def, hz_def


# ----------------------------------------------------------------------------------------------- màu sắc / hiệu ứng
def harmonize(patch, mask, bg_region, strength):
    """Chuyển thống kê màu LAB của vật về gần vùng nền (Reinhard) với cường độ strength."""
    if strength <= 0 or bg_region.size == 0:
        return patch
    pl = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    bl = cv2.cvtColor(bg_region, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    m = mask.astype(bool)
    if m.sum() < 10:
        return patch
    ps = pl[m]
    pm, pstd = ps.mean(0), ps.std(0) + 1e-3
    bm, bstd = bl.mean(0), bl.std(0) + 1e-3
    ratio = np.clip(bstd / pstd, 0.5, 1.6)
    tgt = (pl - pm) * ratio + bm
    # kênh L đổi mạnh (ngày -> đêm), kênh màu đổi nhẹ hơn để không bị bệt màu
    w = np.array([strength, strength * 0.35, strength * 0.35], np.float32)
    out = pl * (1 - w) + tgt * w
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def match_sharpness(patch, bg_region):
    def lv(x):
        return cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if bg_region.size == 0 or min(patch.shape[:2]) < 8:
        return patch
    r = lv(patch) / max(lv(bg_region), 1e-3)
    if r > 2.5:
        return cv2.GaussianBlur(patch, (0, 0), 0.8 if r < 6 else 1.2)
    return patch


def add_noise_like(patch, bg_region, rng):
    if bg_region.size == 0:
        return patch
    g = cv2.cvtColor(bg_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = float(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.0)))
    if noise < 2:
        return patch
    n = rng.normal(0, noise * 0.7, patch.shape).astype(np.float32)
    return np.clip(patch.astype(np.float32) + n, 0, 255).astype(np.uint8)


def rain_fx(img, rng):
    H, W = img.shape[:2]
    drops = np.zeros((H, W), np.float32)
    n = int(H * W * rng.uniform(0.0015, 0.004))
    ys, xs = rng.integers(0, H, n), rng.integers(0, W, n)
    drops[ys, xs] = rng.uniform(0.6, 1.0, n)
    L = int(rng.integers(max(9, H // 40), max(12, H // 18)))
    ang = rng.uniform(-15, 15)
    k = np.zeros((L, L), np.float32)
    k[:, L // 2] = 1.0
    M = cv2.getRotationMatrix2D((L / 2 - 0.5, L / 2 - 0.5), ang, 1.0)
    k = cv2.warpAffine(k, M, (L, L))
    k /= max(k.sum(), 1e-6)
    streak = cv2.filter2D(drops, -1, k) * L * 0.55
    streak = np.clip(streak, 0, 1)[..., None]
    out = img.astype(np.float32)
    out = out * 0.88 + 12  # trời mưa: giảm tương phản, mờ hơi nước
    out = out * (1 - streak * 0.55) + 215 * streak * 0.55
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    return np.clip(out, 0, 255).astype(np.uint8)


def night_fx(img, rng):
    f = img.astype(np.float32) / 255.0
    f = np.power(f, rng.uniform(1.6, 2.2)) * rng.uniform(0.45, 0.6)
    f[..., 0] *= 1.15  # ám xanh
    f[..., 2] *= 0.85
    f += rng.normal(0, 0.012, f.shape)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def luminance(img, cbox):
    x0, y0, x1, y1 = cbox
    return float(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).mean() / 255.0)


# ----------------------------------------------------------------------------------------------- dán 1 vật
def try_place(item, cls, img, insts, occ, cbox, ppm_model, args, rng):
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = cbox
    ch, cw = cy1 - cy0, cx1 - cx0
    ph, pw = item['mask'].shape
    a, hz = ppm_model
    for _ in range(args.tries):
        if cls in ELEVATED:
            yb = rng.uniform(cy0 + 0.10 * ch, cy0 + 0.50 * ch)
            th = rng.uniform(0.025, 0.07) * ch
        else:
            lo = max(hz + 0.05 * ch, cy0 + 0.50 * ch)
            yb = rng.uniform(lo, cy0 + 0.98 * ch)
            th = a * (yb - hz) * PRIOR_HEIGHT_M.get(cls, 1.5) * rng.uniform(0.9, 1.1)
        s = th / ph
        if s > args.max_up or s < 0.15:
            continue
        nh, nw = max(2, int(round(ph * s))), max(2, int(round(pw * s)))
        if nh < args.min_h or nw > 0.6 * cw or nh > 0.8 * ch:
            continue
        y0 = int(round(yb)) - nh
        if y0 < cy0 or y0 + nh > cy1 or cw - nw <= 0:
            continue
        x0 = int(rng.integers(cx0, cx1 - nw + 1))
        m = cv2.resize(item['mask'].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.sum() < 30:
            continue
        # không che quá nhiều vật cũ
        sub = occ[y0:y0 + nh, x0:x0 + nw]
        ids, counts = np.unique(sub[m], return_counts=True)
        bad = False
        for i_, n_ in zip(ids, counts):
            if i_ < 0:
                continue
            if n_ / max(1, insts[i_].area) > args.max_occlude:
                bad = True
                break
            # đúng chiều sâu: vật cũ GẦN camera hơn (đáy thấp hơn) thì phải che vật dán, không được dán đè lên nó
            if cls in GROUND and insts[i_].cls in GROUND and insts[i_].box[3] > y0 + nh + 2:
                bad = True
                break
        if bad:
            continue
        return x0, y0, nw, nh, m, s
    return None


def paste(img, item, x0, y0, nw, nh, m, args, rng):
    rgb = item['rgb']
    patch = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if nw < rgb.shape[1] else cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    pad = max(nh, nw)
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(W, x0 + nw + pad), min(H, y0 + nh + pad)
    region = img[ry0:ry1, rx0:rx1]
    patch = harmonize(patch, m, region, args.harmonize)
    patch = match_sharpness(patch, region)
    patch = add_noise_like(patch, region, rng)
    alpha = m.astype(np.float32)
    if min(nh, nw) >= 12:
        # viền mềm CHỈ phía trong mask (không để lộ nền của ảnh nguồn ra ngoài)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0) * m
    alpha = alpha[..., None]
    dst = img[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
    img[y0:y0 + nh, x0:x0 + nw] = np.clip(alpha * patch + (1 - alpha) * dst, 0, 255).astype(np.uint8)
    return m


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', nargs='+', required=True, help='Dataset 11 lớp chứa instance lớp hiếm để cắt')
    ap.add_argument('--bg', nargs='+', required=True, help='Dataset 11 lớp làm nền (nhãn phải ĐẦY ĐỦ)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', default='3,6,7', help='Lớp được nhân bản')
    ap.add_argument('--n', type=int, default=300, help='Số ảnh tổng hợp cần tạo')
    ap.add_argument('--per-image', type=int, nargs=2, default=[1, 3], metavar=('MIN', 'MAX'))
    ap.add_argument('--bg-mode', default='any', choices=['any', 'night', 'day'],
                    help='night = chỉ nền tối (tự đo độ sáng). Nền mưa: đưa thư mục ảnh mưa vào --bg')
    ap.add_argument('--night-thr', type=float, default=0.28, help='Độ sáng trung bình < ngưỡng => ảnh đêm')
    ap.add_argument('--night-fx', type=float, default=0.0, help='Xác suất biến nền ngày thành đêm giả lập')
    ap.add_argument('--rain-fx', type=float, default=0.0, help='Xác suất thêm vệt mưa giả lập')
    ap.add_argument('--harmonize', type=float, default=0.5, help='Cường độ hoà màu 0..1 (độ sáng theo nền, màu sắc giữ phần lớn)')
    ap.add_argument('--max-up', type=float, default=1.6, help='Phóng to vật tối đa (tránh vỡ hạt)')
    ap.add_argument('--max-occlude', type=float, default=0.5, help='Tỉ lệ tối đa được che 1 vật cũ')
    ap.add_argument('--min-h', type=int, default=20, help='Chiều cao tối thiểu vật sau khi dán (px)')
    ap.add_argument('--max-reuse', type=int, default=12,
                    help='Mỗi instance nguồn được dán tối đa N lần (chống học thuộc khi nguồn ít)')
    ap.add_argument('--tries', type=int, default=40)
    ap.add_argument('--golden-val', default=None, help='Ảnh golden_val: loại nguồn/nền trùng')
    ap.add_argument('--seed', type=int, default=27)
    ap.add_argument('--quality', type=int, default=93)
    ap.add_argument('--viz', type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prng = random.Random(args.seed)
    classes = parse_int_list(args.classes)
    gold = HashIndex.from_dir(args.golden_val)
    t0 = time.time()
    bank = build_bank(args.src, classes, gold)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    bank_preview(bank, Path(args.out) / 'bank_preview.jpg')
    avail = [c for c in classes if bank[c]]
    if not avail:
        sys.exit('Ngân hàng instance rỗng: dữ liệu --src không có lớp ' + args.classes +
                 ' đủ chuẩn (nguyên vẹn, cao >= 24px). Hãy thêm nguồn (COCO/autolabel) rồi chạy lại.')
    for c in avail:
        if len(bank[c]) < 20:
            print(f'!!! Lớp {c} {CLASSES[c]} chỉ có {len(bank[c])} instance nguồn: dán lại nhiều lần dễ làm model HỌC THUỘC. '
                  f'Nên bổ sung nguồn (COCO/OpenImages/autolabel). Mỗi instance tối đa --max-reuse={args.max_reuse} lần.')
    uses = {c: np.zeros(len(bank[c]), int) for c in avail}

    def pick_class():
        # ưu tiên lớp hiếm hơn (trọng số 1/sqrt(n)), bỏ lớp đã dùng hết quota
        live = [c for c in avail if (uses[c] < args.max_reuse).any()]
        if not live:
            return None
        w = np.array([1 / np.sqrt(len(bank[c])) for c in live])
        return int(rng.choice(live, p=w / w.sum()))

    # nền
    bgs = []
    for root in args.bg:
        for p in list_images(root):
            bgs.append((p, Path(root)))
    prng.shuffle(bgs)
    out = Path(args.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    manifest = open(out / 'manifest.jsonl', 'w', encoding='utf-8')
    made, tried_bg, rej_bright, rej_leak = 0, 0, 0, 0
    pasted_cnt = {CLASSES[c]: 0 for c in classes}
    viz = []
    bi = 0
    while made < args.n and tried_bg < len(bgs) * 3 and pick_class() is not None:
        p, root = bgs[bi % len(bgs)]
        bi += 1
        tried_bg += 1
        img = imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        cbox = content_box(img)
        if (cbox[2] - cbox[0]) < 0.3 * W or (cbox[3] - cbox[1]) < 0.3 * H:
            continue
        lum = luminance(img, cbox)
        if args.bg_mode == 'night' and lum >= args.night_thr:
            rej_bright += 1
            continue
        if args.bg_mode == 'day' and lum < args.night_thr:
            rej_bright += 1
            continue
        if gold.names and gold.nearest(phash(img))[0] <= 4:
            rej_leak += 1
            continue
        fx = []
        if args.night_fx > 0 and lum >= args.night_thr and rng.random() < args.night_fx:
            x0c, y0c, x1c, y1c = cbox
            img[y0c:y1c, x0c:x1c] = night_fx(np.ascontiguousarray(img[y0c:y1c, x0c:x1c]), rng)
            fx.append('night_fx')
        lp = find_label(p, root)
        insts = []
        for c, pts, kind in read_label_file(lp):
            it = Inst.from_polygon(pts * [W, H], H, W, cls=c)
            if it is not None:
                it.line = format_line(c, pts)
                insts.append(it)
        # bản đồ chiếm chỗ (id instance theo pixel, xét chồng lấn hiện có)
        occ = np.full((H, W), -1, np.int32)
        for k, it in enumerate(insts):
            x0, y0, x1, y1 = it.box
            occ[y0:y1, x0:x1][it.mask] = k
        ppm_model = fit_ppm(insts, cbox)
        n_paste = int(rng.integers(args.per_image[0], args.per_image[1] + 1))
        added = []
        for _ in range(n_paste):
            c = pick_class()
            if c is None:
                break
            free = np.flatnonzero(uses[c] < args.max_reuse)
            j = int(free[rng.integers(len(free))])
            item = bank[c][j]
            flip = c != SIGN and rng.random() < 0.5  # biển báo có chữ/ký hiệu: không lật
            if flip:
                item = dict(item, rgb=np.ascontiguousarray(item['rgb'][:, ::-1]), mask=np.ascontiguousarray(item['mask'][:, ::-1]))
            pl = try_place(item, c, img, insts, occ, cbox, ppm_model, args, rng)
            if pl is None:
                continue
            x0, y0, nw, nh, m, s = pl
            m = paste(img, item, x0, y0, nw, nh, m, args, rng)
            new = Inst.from_mask(m, x0, y0, cls=c, src='paste')
            if new is None:
                continue
            insts.append(new)
            uses[c][j] += 1
            k = len(insts) - 1
            occ[y0:y0 + nh, x0:x0 + nw][m] = k
            added.append(dict(cls=CLASSES[c], src=item['src'], box=[x0, y0, x0 + nw, y0 + nh], scale=round(s, 3), flip=bool(flip)))
            pasted_cnt[CLASSES[c]] += 1
        if not added:
            continue
        if args.rain_fx > 0 and rng.random() < args.rain_fx:
            x0c, y0c, x1c, y1c = cbox  # chỉ vùng ảnh thật, không vẽ mưa lên viền đen letterbox
            img[y0c:y1c, x0c:x1c] = rain_fx(np.ascontiguousarray(img[y0c:y1c, x0c:x1c]), rng)
            fx.append('rain_fx')
        # Zero-Overlap: vật dán (thêm sau) luôn ở trên cùng -> dùng chính bản đồ occ
        lines = []
        for k, it in enumerate(insts):
            x0, y0, x1, y1 = it.box
            vis = occ[y0:y1, x0:x1] == k
            a1 = int(vis.sum())
            if a1 == 0:
                continue
            if a1 >= it.area - max(3, 0.002 * it.area) and it.line is not None:  # không bị che -> giữ polygon gốc
                lines.append(it.line)
                continue
            v = Inst.from_mask(vis, x0, y0, cls=it.cls)
            if v is None or too_small(v, H, W):
                continue
            ln = v.to_line(H, W)
            if ln:
                lines.append(ln)
        name = f'cp{made:05d}_{p.stem[:50]}'
        imwrite(out / 'images' / f'{name}.jpg', img, args.quality)
        (out / 'labels' / f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        manifest.write(json.dumps(dict(image=f'{name}.jpg', background=str(p), bg_luminance=round(lum, 3), fx=fx,
                                       pasted=added), ensure_ascii=False) + '\n')
        if len(viz) < args.viz:
            viz.append((out / 'images' / f'{name}.jpg', out / 'labels' / f'{name}.txt'))
        made += 1
        if made % 50 == 0:
            print(f'  {made}/{args.n} ảnh ({(time.time() - t0) / 60:.1f} phút)', flush=True)
    manifest.close()
    write_data_yaml(out)
    save_json(out / 'copy_paste_log.json', dict(
        tool='tools_tv2/copy_paste.py', time=now(), src=args.src, bg=args.bg, classes=[CLASSES[c] for c in classes],
        bank_size={CLASSES[c]: len(bank[c]) for c in classes}, n_made=made, pasted=pasted_cnt,
        bg_mode=args.bg_mode, night_thr=args.night_thr, rejected_brightness=rej_bright, rejected_golden_leak=rej_leak,
        params={k: v for k, v in vars(args).items() if k not in ('src', 'bg')}))
    if viz:
        from visualize import draw_pair_grid  # noqa: PLC0415
        draw_pair_grid(viz, out / 'viz')
    print(f'\nĐã tạo {made} ảnh tổng hợp trong {out} ({(time.time() - t0) / 60:.1f} phút). Đã dán: ' +
          ', '.join(f'{k}={v}' for k, v in pasted_cnt.items()))
    if made < args.n:
        print(f'!!! Chỉ tạo được {made}/{args.n}: hết ảnh nền phù hợp (loại do độ sáng: {rej_bright}, '
              f'trùng golden_val: {rej_leak}) hoặc hết quota --max-reuse của ngân hàng instance. '
              'Thêm ảnh nền / nguồn, hoặc dùng --bg-mode any / --night-fx.')


if __name__ == '__main__':
    main()
