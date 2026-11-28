"""Luồng xử lý 1 ảnh: nhãn có sẵn + phát hiện mới -> nhãn 11 lớp sạch, đúng quy tắc cuộc thi."""
from __future__ import annotations

import numpy as np

from .core import (CLASSES, GROUP, OTHER, PED, TRUCK, TWO_WHEEL, VEHICLES, Inst, box_ioa, box_iou, content_box,
                   is_ego_hood, mask_inter, resolve_overlaps, sanitize_inst, too_small)
from .pairing import pair_riders, remove_people_in_vehicles


def _rv(review, reason, inst=None, **kw):
    row = dict(reason=reason)
    if inst is not None:
        row.update(cls=inst.cls, box=inst.box, conf=round(float(inst.conf), 3), src=inst.src)
    row.update(kw)
    review.append(row)


DEDUPE_THR = {'veh': (0.45, 0.7), 'tw': (0.30, 0.5), 'ped': (0.5, 0.7), 'sign': (0.4, 0.6), 'light': (0.4, 0.6)}


def dedupe_against(existing, new):
    """Bỏ phát hiện mới trùng với nhãn có sẵn (cùng nhóm lớp: xe 4 bánh / xe 2 bánh / người / biển / đèn).
    Xe 2 bánh dùng ngưỡng lỏng hơn vì nhãn người vẽ thường chỉ phủ thân xe, còn hộp model phủ cả người lái."""
    keep = []
    for d in new:
        dup = False
        for e in existing:
            g = GROUP[e.cls]
            if g != GROUP[d.cls]:
                continue
            iou_thr, ioa_thr = DEDUPE_THR[g]
            if box_iou(d.box, e.box) >= iou_thr or box_ioa(d.box, e.box) >= ioa_thr or box_ioa(e.box, d.box) >= 0.85:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def split_trailers(insts, trailer_dets, review, enable):
    """Tách rơ-moóc/container khỏi đầu kéo: container nằm trong hộp Truck -> container = 3, Truck -= container."""
    out = list(insts)
    for c in trailer_dets:
        hosts = [t for t in out if t.cls == TRUCK and box_ioa(c.box, t.box) >= 0.6]
        if not hosts:
            if enable:
                out.append(c)
                _rv(review, 'trailer_standalone_added', c)
            continue
        host = max(hosts, key=lambda t: box_ioa(c.box, t.box))
        tb, cb = host.box, c.box
        area_ratio = ((cb[2] - cb[0]) * (cb[3] - cb[1])) / max(1, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        if not enable:
            _rv(review, 'truck_maybe_has_trailer_check_split', host)
            continue
        if area_ratio > 0.9:
            # container phủ gần hết xe: thường là xe tải thùng (vẫn là Truck) hoặc nhìn từ phía sau -> người quyết
            _rv(review, 'truck_or_trailer_ambiguous', host)
            continue
        x0, y0, x1, y1 = tb
        rest = host.crop(x0, y0, x1, y1) & ~c.crop(x0, y0, x1, y1)
        new_truck = Inst.from_mask(rest, x0, y0, cls=TRUCK, conf=host.conf, src=host.src, meta=dict(host.meta))
        if new_truck is None or new_truck.area < 0.1 * host.area:
            _rv(review, 'truck_or_trailer_ambiguous', host)
            continue
        out[out.index(host)] = new_truck
        c.cls = OTHER
        out.append(c)
        _rv(review, 'trailer_split_from_truck', c)
    return out


def process_image(H, W, existing, new_insts, cfg, review, img=None, refine_fn=None, trailer_dets=()):
    """existing: Inst từ nhãn có sẵn (src='human'); new_insts: Inst từ model (đã có mask).
    Trả về list Inst cuối cùng (không chồng lấn, đã lọc)."""
    cbox = content_box(img) if img is not None else (0, 0, W, H)

    # 1) làm sạch nhãn có sẵn
    ex = []
    for e in existing:
        if cfg.get('drop_ego_hood', True) and is_ego_hood(e, cbox):
            _rv(review, 'ego_hood_dropped', e)
            continue
        if cfg.get('sanitize', True):
            e2, changed = sanitize_inst(e)
            if changed:
                _rv(review, 'human_polygon_spike_removed', e2)
            e = e2
        if e.meta.get('was_box'):
            _rv(review, 'bbox_line_converted_to_rectangle_polygon', e)
        ex.append(e)

    # 2) phát hiện mới: lọc nhỏ, nắp capo, trùng nhãn có sẵn
    nw = []
    for d in new_insts:
        if too_small(d, H, W):
            continue
        if cfg.get('drop_ego_hood', True) and is_ego_hood(d, cbox):
            continue
        nw.append(d)
    nw = dedupe_against(ex, nw)

    # open-vocab "Other vehicle" trùng xe COCO -> giữ lớp COCO, ghi review
    vehicles_known = [i for i in ex + nw if i.cls in VEHICLES | TWO_WHEEL and i.src != 'ov']
    final_new = []
    for d in nw:
        if d.src == 'ov' and d.cls == OTHER:
            clash = [v for v in vehicles_known if box_iou(d.box, v.box) >= 0.45 or box_ioa(v.box, d.box) >= 0.8]
            if clash:
                _rv(review, 'maybe_other_vehicle_check_class', clash[0], ov_name=d.meta.get('name', ''))
                continue
        final_new.append(d)
    insts = ex + final_new
    for d in final_new:
        if d.conf < cfg.get('review_conf', 0.5):
            _rv(review, 'low_conf_auto_label', d, name=d.meta.get('name', ''))

    # 3) rơ-moóc / container
    trailer = [t for t in trailer_dets if not too_small(t, H, W)]
    if trailer:
        insts = split_trailers(insts, trailer, review, cfg.get('split_trailer', False))

    # 4) người trong ô tô -> bỏ ; 5) ghép người + xe 2 bánh
    insts = remove_people_in_vehicles(insts, review)
    if cfg.get('pairing', True):
        insts, review = pair_riders(insts, cfg.get('pair_cfg', {}), review, refine_fn=refine_fn)
    # người chỉ dùng để ghép cặp (lớp 8 không nằm trong --add-classes) mà không ngồi xe -> bỏ
    insts = [i for i in insts if not (i.cls == PED and i.meta.get('helper'))]

    # 6) Zero-Overlap + lọc cuối
    res = resolve_overlaps(insts, H, W)
    out = []
    for (vis, lost), orig in zip(res, insts):
        if vis is None:
            continue
        if orig.src != 'human' and lost > 0.75:
            _rv(review, 'auto_instance_mostly_hidden_dropped', orig, lost=round(lost, 2))
            continue
        if orig.cls == PED and lost > 0.6 and orig.src != 'human':
            continue  # người bị che gần hết bởi xe máy có người: thường là nhận trùng người ngồi sau
        if too_small(vis, H, W):
            continue
        out.append(vis)
    return out


def stats(insts):
    c = [0] * len(CLASSES)
    for i in insts:
        c[i.cls] += 1
    return c


def priority(review_rows):
    """Điểm ưu tiên duyệt (active learning): càng nhiều ca nghi ngờ / conf thấp -> duyệt trước."""
    w = {'pair_ambiguous_2_vehicles': 3, 'pair_borderline': 2, 'person_near_two_wheeler_kept_as_pedestrian': 2,
         'truck_or_trailer_ambiguous': 3, 'trailer_split_from_truck': 2, 'truck_maybe_has_trailer_check_split': 2,
         'maybe_other_vehicle_check_class': 2, 'low_conf_auto_label': 1, 'person_in_vehicle_dropped': 1,
         'ego_hood_dropped': 0.5, 'human_polygon_spike_removed': 0.5, 'bbox_line_converted_to_rectangle_polygon': 2,
         'person_on_other_vehicle_merged': 2, 'trailer_standalone_added': 2, 'sam_failed': 2,
         'auto_instance_mostly_hidden_dropped': 0.5}
    return sum(w.get(r['reason'], 1) for r in review_rows)


__all__ = ['process_image', 'stats', 'priority', 'mask_inter', 'np']
