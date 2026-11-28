#!/usr/bin/env python3
"""[TV2] Gán nhãn nháp tự động: Detector mạnh (YOLO11x / YOLOv8x / best.pt của đội) + Open-vocab (YOLO-World
hoặc GroundingDINO) + SAM 2  ->  polygon YOLO-seg 11 lớp, đã ghép cặp người-xe (4/5, 6/7), Zero-Overlap, lọc vật nhỏ.

Ví dụ (Colab/Kaggle có GPU):
  # 1) Ảnh mới chưa có nhãn -> nhãn nháp đầy đủ 11 lớp
  python autolabel.py --images /content/raw_frames --out /content/draft_v2

  # 2) Bộ Roboflow 5 lớp có sẵn: GIỮ polygon người vẽ, remap 5 -> 11 lớp, bổ sung người/biển/đèn/xe khác,
  #    tự ghép người ngồi vào xe máy/xe đạp
  python autolabel.py --images "data/Vietnam Traffic Vehicle Detectio.v1i.yolo26" --out data/roboflow_v1_11cls --keep-labels

  # 3) Chỉ ghép cặp + remap, KHÔNG chạy model (CPU, vài giây) -> xem pair_riders.py

Kết quả trong --out:
  <cấu trúc thư mục giống đầu vào>/images, /labels ; data.yaml ; review.csv (ca cần người duyệt, xếp theo ưu tiên)
  run_log.json (tham số, ánh xạ lớp, thống kê — dùng cho slide báo cáo) ; viz/ (ảnh vẽ nhãn để kiểm tra nhanh)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tv2lib.core import (CLASSES, Inst, find_data_yaml, imread, list_images, load_insts, load_src_map,  # noqa: E402
                         label_rel_for, now, out_img_rel, parse_int_list, progress, save_json, write_data_yaml)
from tv2lib.pipeline import priority, process_image, stats  # noqa: E402


def build_parser(no_detect_default=False):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True, help='Thư mục ảnh (quét đệ quy) hoặc gốc dataset (train/valid/test)')
    ap.add_argument('--out', required=True, help='Thư mục kết quả (không ghi đè dữ liệu gốc)')
    ap.add_argument('--keep-labels', action='store_true', help='Giữ nhãn có sẵn (người vẽ), chỉ bổ sung cái còn thiếu')
    ap.add_argument('--src-map', default='auto',
                    help="Ánh xạ lớp của nhãn có sẵn: 'auto' (đọc names trong data.yaml), 'identity' (đã 11 lớp), "
                         "hoặc '0:6,1:1,2:0,3:4,4:2' (gốc:chuẩn, -1=bỏ)")
    ap.add_argument('--src-yaml', default=None, help='data.yaml của bộ gốc (mặc định tự tìm)')
    ap.add_argument('--no-detect', action='store_true', default=no_detect_default,
                    help='Không chạy model: chỉ remap + làm sạch + ghép cặp + zero-overlap (CPU)')
    ap.add_argument('--det-model', default='yolo11x.pt', help='yolo11x.pt | yolov8x.pt | yolo11x-seg.pt | best.pt của đội')
    ap.add_argument('--det-imgsz', type=int, default=1280, help='Chạy detector ở ảnh lớn để bắt vật nhỏ')
    ap.add_argument('--open-vocab', default='world', choices=['world', 'gdino', 'none'],
                    help='Bộ phát hiện open-vocab cho biển báo / xích lô / ba gác / container')
    ap.add_argument('--ov-model', default=None, help='yolov8x-worldv2.pt (world) | IDEA-Research/grounding-dino-base (gdino)')
    ap.add_argument('--sam-model', default='sam2.1_b.pt', help='sam2.1_t/s/b/l.pt (l chính xác nhất, chậm nhất)')
    ap.add_argument('--mask-source', default='sam', choices=['sam', 'det'],
                    help="sam = SAM 2 cho mọi hộp (tốt nhất); det = dùng mask của model -seg (nhanh, cần *-seg.pt)")
    ap.add_argument('--add-classes', default='all', help="Lớp được phép THÊM mới từ model, vd '8,9,10' hoặc 'all'")
    ap.add_argument('--split-trailer', action='store_true',
                    help='Tự tách container/rơ-moóc khỏi đầu kéo (2 -> 2 + 3). Tắt: chỉ đánh dấu trong review.csv')
    ap.add_argument('--no-rider-refine', action='store_true', help='Không dùng SAM trên hộp hợp người+xe (lấy hàng hoá/áo mưa)')
    ap.add_argument('--refine-human-masks', action='store_true',
                    help='(--keep-labels) Dùng SAM vẽ lại mask người vẽ cho sát vật hơn nếu khớp IoU >= --refine-min-iou '
                         '(nên bật cho bộ Roboflow: polygon gốc khá thô/lệch)')
    ap.add_argument('--refine-min-iou', type=float, default=0.5)
    ap.add_argument('--no-pairing', action='store_true')
    ap.add_argument('--no-sanitize', action='store_true', help='Không xoá gai/đường kẻ lỗi trên polygon có sẵn')
    ap.add_argument('--keep-ego-hood', action='store_true', help='Không bỏ nắp capo xe dashcam bị gán nhầm là Car')
    ap.add_argument('--review-conf', type=float, default=0.5, help='Nhãn tự động có conf dưới mức này -> review.csv')
    ap.add_argument('--device', default=None, help="0 | cpu (mặc định tự chọn)")
    ap.add_argument('--limit', type=int, default=0, help='Chỉ chạy N ảnh đầu (thử nhanh)')
    ap.add_argument('--resume', action='store_true', help='Bỏ qua ảnh đã có nhãn trong --out (chạy tiếp khi Colab ngắt)')
    ap.add_argument('--viz', type=int, default=40, help='Số ảnh vẽ nhãn để kiểm tra (0 = tắt)')
    ap.add_argument('--no-copy-images', action='store_true', help='Không chép ảnh sang --out (chỉ ghi nhãn)')
    return ap


class Models:
    def __init__(self, args, log=print):
        from tv2lib import detect  # noqa: PLC0415 (chỉ nạp torch khi cần)
        self.detect = detect
        self.args = args
        self.det = detect.UltraDetector(args.det_model, args.det_imgsz, args.device, log=log)
        self.ov = None
        if args.open_vocab == 'world':
            self.ov = detect.WorldDetector(args.ov_model or 'yolov8x-worldv2.pt', args.det_imgsz, args.device, log=log)
        elif args.open_vocab == 'gdino':
            self.ov = detect.GDinoDetector(args.ov_model or 'IDEA-Research/grounding-dino-base', args.device, log=log)
        self.sam = None
        if args.mask_source == 'sam' or args.open_vocab != 'none' or args.refine_human_masks:
            self.sam = detect.SamSegmenter(args.sam_model, args.device, log=log)

    def run(self, img, review):
        """-> (list Inst thường, list Inst rơ-moóc/container)."""
        H, W = img.shape[:2]
        dets = self.det(img)
        trailer = []
        if self.ov is not None:
            ovd = self.ov(img)
            trailer = [d for d in ovd if d.kind == 'trailer']
            dets = dets + [d for d in ovd if d.kind != 'trailer']
        dets = self.detect.nms_dets(dets, 0.6)
        allow = set(parse_int_list(self.args.add_classes))
        # người luôn được phát hiện (cần cho ghép cặp); nếu lớp 8 không được phép thêm thì người không ngồi xe bị bỏ sau
        dets = [d for d in dets if d.cls in allow or (d.cls in (5, 7) and {4, 6} & allow) or d.cls == 8]
        self.helper_person = 8 not in allow
        # bỏ hộp quá nhỏ trước khi chạy SAM (tiết kiệm thời gian)
        L = max(H, W)
        dets = [d for d in dets if min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= 0.0125 * L]
        trailer = [d for d in trailer if min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= 0.0125 * L]

        insts = []
        need_sam = [d for d in dets if self.args.mask_source == 'sam' or d.poly is None or d.src == 'ov']
        sam_masks = dict(zip(map(id, need_sam), self.sam(img, [d.box for d in need_sam]))) if (self.sam and need_sam) else {}
        for d in dets:
            m = sam_masks.get(id(d))
            bw, bh = d.box[2] - d.box[0], d.box[3] - d.box[1]
            ok = m is not None and m.area >= 0.08 * bw * bh
            if not ok and d.poly is not None:
                m = Inst.from_polygon(d.poly, H, W, cls=d.cls)
                ok = m is not None
            if not ok:
                review.append(dict(reason='sam_failed', cls=d.cls, box=tuple(int(x) for x in d.box), conf=round(d.conf, 3)))
                continue
            m.cls, m.conf, m.src = d.cls, d.conf, d.src
            m.meta = {'name': d.name}
            if d.cls == 8 and self.helper_person:
                m.meta['helper'] = True
            insts.append(m)
        tr = []
        if trailer and self.sam:
            for d, m in zip(trailer, self.sam(img, [d.box for d in trailer])):
                if m is not None:
                    m.cls, m.conf, m.src, m.meta = 3, d.conf, 'ov', {'name': d.name}
                    tr.append(m)
        return insts, tr

    def refine_existing(self, img, insts, review):
        """Thay mask người vẽ (thường thô/lệch) bằng mask SAM bám sát vật hơn — chỉ khi khớp tốt (IoU >= 0.5)."""
        if self.sam is None or not insts:
            return insts, 0
        L = max(img.shape[:2])
        idx = [k for k, e in enumerate(insts) if min(e.box[2] - e.box[0], e.box[3] - e.box[1]) >= max(12, 0.02 * L)]
        if not idx:
            return insts, 0
        masks = self.sam(img, [insts[k].box for k in idx], pad=0.05)
        n = 0
        out = list(insts)
        from tv2lib.core import mask_inter  # noqa: PLC0415
        for k, m in zip(idx, masks):
            e = insts[k]
            if m is None:
                continue
            inter = mask_inter(e, m)
            iou = inter / max(1, e.area + m.area - inter)
            if iou >= self.args.refine_min_iou and 0.6 <= m.area / max(1, e.area) <= 1.6:
                m.cls, m.conf, m.src, m.meta = e.cls, e.conf, 'human', dict(e.meta, sam_refined=round(iou, 3))
                out[k] = m
                n += 1
        return out, n

    def refine_fn(self, img):
        if self.sam is None or self.args.no_rider_refine:
            return None
        return lambda boxes: self.sam(img, boxes, pad=0.05)


def run(args, log=print):
    t_start = time.time()
    in_root = Path(args.images)
    out_root = Path(args.out)
    if out_root.resolve() == in_root.resolve():
        sys.exit('--out phải khác --images (không ghi đè dữ liệu gốc)')
    imgs = list_images(in_root)
    if args.limit:
        imgs = imgs[:args.limit]
    if not imgs:
        sys.exit(f'Không thấy ảnh nào trong {in_root}')
    log(f'[{now()}] {len(imgs)} ảnh trong {in_root}')

    src_map, map_mode = ({}, 'none')
    if args.keep_labels:
        yml = args.src_yaml or find_data_yaml(in_root)
        src_map, map_mode = load_src_map(args.src_map, yml, log)

    models = None if args.no_detect else Models(args, log)
    cfg = dict(drop_ego_hood=not args.keep_ego_hood, sanitize=not args.no_sanitize, pairing=not args.no_pairing,
               split_trailer=args.split_trailer, review_conf=args.review_conf)

    out_root.mkdir(parents=True, exist_ok=True)
    review_all, per_img = [], []
    tot_in, tot_out = [0] * len(CLASSES), [0] * len(CLASSES)
    viz_list = []
    t0 = time.time()
    n_done = n_skip = n_fail = 0
    n_refined = 0
    for k, p in enumerate(imgs, 1):
        rel = p.relative_to(in_root)
        o_img = out_root / out_img_rel(rel)
        o_lbl = out_root / label_rel_for(out_img_rel(rel))
        if args.resume and o_lbl.is_file():
            n_skip += 1
            continue
        img = imread(p)
        if img is None:
            log(f'  ! không đọc được {p}')
            n_fail += 1
            continue
        H, W = img.shape[:2]
        review = []
        existing = []
        if args.keep_labels:
            existing, _ = load_insts(p, in_root, H, W, src_map)
        tot_in = [a + b for a, b in zip(tot_in, stats(existing))]
        if models is not None and existing and args.refine_human_masks:
            existing, nref = models.refine_existing(img, existing, review)
            n_refined += nref
        new, trailer = ([], []) if models is None else models.run(img, review)
        refine = None if models is None else models.refine_fn(img)
        final = process_image(H, W, existing, new, cfg, review, img=img, refine_fn=refine, trailer_dets=trailer)
        lines = [ln for ln in (i.to_line(H, W) for i in final) if ln]
        o_lbl.parent.mkdir(parents=True, exist_ok=True)
        o_lbl.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        if not args.no_copy_images:
            o_img.parent.mkdir(parents=True, exist_ok=True)
            if not o_img.exists():
                shutil.copy2(p, o_img)
        tot_out = [a + b for a, b in zip(tot_out, stats(final))]
        rel_s = str(out_img_rel(rel)).replace('\\', '/')
        for r in review:
            r['image'] = rel_s
            review_all.append(r)
        per_img.append((priority(review), rel_s, len(review)))
        if args.viz and len(viz_list) < args.viz and (review or k % max(1, len(imgs) // max(1, args.viz)) == 0):
            viz_list.append((o_img if o_img.exists() else p, o_lbl))
        n_done += 1
        progress(k, len(imgs), t0, prefix='  ')

    # review.csv: sắp theo ưu tiên ảnh
    pr = {name: s for s, name, _ in per_img}
    review_all.sort(key=lambda r: (-pr.get(r['image'], 0), r['image']))
    with open(out_root / 'review.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 'priority', 'reason', 'class_id', 'class', 'box_x1y1x2y2', 'conf', 'src', 'extra'])
        for r in review_all:
            c = r.get('cls', '')
            extra = {k: v for k, v in r.items() if k not in ('image', 'reason', 'cls', 'box', 'conf', 'src')}
            w.writerow([r['image'], pr.get(r['image'], 0), r['reason'], c, CLASSES[c] if c != '' else '',
                        ' '.join(str(int(x)) for x in r.get('box', ())), r.get('conf', ''), r.get('src', ''),
                        extra or ''])
    with open(out_root / 'review_priority.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 'priority', 'n_flags'])
        for s, name, n in sorted(per_img, reverse=True):
            if n:
                w.writerow([name, s, n])

    splits = {}
    for s in ('train', 'valid', 'val', 'test'):
        if (out_root / s / 'images').is_dir():
            splits['val' if s in ('valid', 'val') and 'val' not in splits else s] = f'{s}/images'
    write_data_yaml(out_root, splits or None)

    reasons = {}
    for r in review_all:
        reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    log_obj = dict(
        tool='tools_tv2/autolabel.py', time=now(), minutes=round((time.time() - t_start) / 60, 2),
        input=str(in_root), output=str(out_root), n_images=len(imgs), n_done=n_done, n_skipped_resume=n_skip,
        n_failed=n_fail, keep_labels=args.keep_labels, human_masks_refined_by_sam=n_refined, class_map_mode=map_mode,
        class_map={str(k): (CLASSES[v] if v >= 0 else 'DROP') for k, v in src_map.items()},
        models=None if args.no_detect else dict(det=args.det_model, det_imgsz=args.det_imgsz, open_vocab=args.open_vocab,
                                                ov_model=args.ov_model, sam=args.sam_model, mask_source=args.mask_source),
        config=cfg, add_classes=args.add_classes,
        instances_before={CLASSES[i]: tot_in[i] for i in range(len(CLASSES))},
        instances_after={CLASSES[i]: tot_out[i] for i in range(len(CLASSES))},
        review_reasons=reasons,
    )
    save_json(out_root / 'run_log.json', log_obj)

    if viz_list:
        from visualize import draw_pair_grid  # noqa: PLC0415
        draw_pair_grid(viz_list, out_root / 'viz')

    if args.no_detect and tot_in[8] == 0 and (tot_in[4] + tot_in[6]) > 0:
        log('\n!!! CẢNH BÁO: dữ liệu gốc KHÔNG có nhãn người (Pedestrian) nên KHÔNG ghép cặp được: mọi xe máy/xe đạp '
            'đang để lớp 4/6 (không người) — SAI với xe đang có người lái. Hãy chạy autolabel.py --keep-labels '
            '(có model tìm người, cần GPU) thay vì chế độ --no-detect / pair_riders.py.')
    log('\n========== TỔNG KẾT ==========')
    log(f'{"lớp":<15}{"trước":>8}{"sau":>8}')
    for i, c in enumerate(CLASSES):
        log(f'{i:>2} {c:<12}{tot_in[i]:>8}{tot_out[i]:>8}')
    log(f'Ảnh xử lý: {n_done}, bỏ qua (resume): {n_skip}, lỗi đọc: {n_fail}, mask người vẽ được SAM làm sạch: {n_refined}')
    log(f'Cờ cần duyệt: {len(review_all)} -> {out_root / "review.csv"} (đã xếp ảnh cần xem trước lên đầu)')
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        log(f'   {v:>6}  {k}')
    log(f'Xong sau {(time.time() - t_start) / 60:.1f} phút. Nhãn ở {out_root}')
    return log_obj


if __name__ == '__main__':
    run(build_parser().parse_args())
