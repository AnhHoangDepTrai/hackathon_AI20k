"""Ghép cặp Người <-> Xe 2 bánh (và xích lô/ba gác) để tách lớp 4/5 và 6/7.

Quy tắc cuộc thi:
  - Có người NGỒI trên xe  -> gộp người + xe thành MỘT instance 5 (Motorcyclist) / 7 (Cyclist)
  - Người DẮT xe / đứng cạnh -> người = 8 (Pedestrian), xe = 4 (Motorcycle) / 6 (Bicycle), hai instance riêng
  - Người ngồi trong ô tô (thấy qua kính) -> KHÔNG gán

Thuật toán (chạy trên bbox + mask, không cần model):
  Với mỗi cặp (người p, xe v) tính các đặc trưng hình học:
    hov      = độ chồng theo chiều ngang / chiều rộng hộp hẹp hơn  (người ngồi: ~1)
    dcx      = lệch tâm ngang / chiều rộng hộp rộng hơn            (người ngồi: nhỏ; người dắt: lớn)
    legs_in  = đáy người nằm sâu bao nhiêu trong hộp xe (0=đỉnh xe, 1=đáy xe)  (người ngồi: 0.3..1.3)
    head     = đỉnh người cao hơn đỉnh xe bao nhiêu (theo chiều cao xe)
    scale    = cao người / cao xe
    ioa_p    = phần hộp người nằm trong hộp xe
    contact  = mask người (nới vài px) chạm mask xe
  Ngồi trên xe nếu thoả tất cả điều kiện; điểm s dùng để chọn xe tốt nhất cho mỗi người (mỗi người 1 xe,
  mỗi xe tối đa N người). Ca sát ngưỡng / mơ hồ được ghi vào review để người duyệt xem trước.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import (BIKE, CAR, BUS, TRUCK, CYCLIST, MOTO, MOTORIST, OTHER, PED, Inst, box_ioa, mask_inter,
                   union_insts)

RIDEABLE = {MOTO: MOTORIST, MOTORIST: MOTORIST, BIKE: CYCLIST, CYCLIST: CYCLIST, OTHER: OTHER}
CAPACITY = {MOTO: 4, MOTORIST: 4, BIKE: 2, CYCLIST: 2, OTHER: 3}
CLOSED_VEHICLES = {CAR, BUS, TRUCK}


def _contact(p: Inst, v: Inst, r):
    """Mask người nới r px có chạm mask xe không (tính cục bộ cho nhanh)."""
    x0, y0 = min(p.box[0], v.box[0]) - r, min(p.box[1], v.box[1]) - r
    x1, y1 = max(p.box[2], v.box[2]) + r, max(p.box[3], v.box[3]) + r
    pm = p.crop(x0, y0, x1, y1).astype(np.uint8)
    vm = v.crop(x0, y0, x1, y1)
    k = 2 * r + 1
    pd = cv2.dilate(pm, np.ones((k, k), np.uint8)).astype(bool)
    return int((pd & vm).sum())


def features(p: Inst, v: Inst):
    px1, py1, px2, py2 = p.box
    vx1, vy1, vx2, vy2 = v.box
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    vw, vh = max(1, vx2 - vx1), max(1, vy2 - vy1)
    ix = max(0, min(px2, vx2) - max(px1, vx1))
    iy = max(0, min(py2, vy2) - max(py1, vy1))
    inter = ix * iy
    return dict(
        hov=ix / min(pw, vw),
        dcx=abs((px1 + px2) / 2 - (vx1 + vx2) / 2) / max(pw, vw),
        legs_in=(py2 - vy1) / vh,
        head=(vy1 - py1) / vh,
        scale=ph / vh,
        ioa_p=inter / (pw * ph),
        ioa_v=inter / (vw * vh),
        vh=vh,
    )


def is_riding(f, contact_px, cfg):
    """Trả về (ngồi_trên_xe?, điểm, lý_do_trượt)."""
    if f['hov'] < cfg.get('min_hov', 0.45):
        return False, 0.0, 'lệch ngang'
    if f['dcx'] > cfg.get('max_dcx', 0.40):
        return False, 0.0, 'lệch tâm'
    if not (0.30 <= f['legs_in'] <= 1.35):
        return False, 0.0, 'chân không nằm trong xe'
    if f['head'] < -0.20:
        return False, 0.0, 'đầu thấp hơn xe'
    if not (0.45 <= f['scale'] <= 4.0):
        return False, 0.0, 'tỉ lệ người/xe bất thường'
    if f['ioa_p'] < 0.12:
        return False, 0.0, 'người ít chồng lên xe'
    if contact_px <= 0 and f['ioa_p'] < 0.30:
        return False, 0.0, 'mask không chạm'
    # người đứng cao hẳn, chân chạm đất cạnh xe, lệch tâm => đang dắt xe
    if f['legs_in'] >= 0.92 and f['dcx'] > 0.26 and f['scale'] >= 1.55:
        return False, 0.0, 'dáng đứng cạnh xe (dắt)'
    s = (0.35 * min(f['ioa_v'] / 0.5, 1.0) + 0.25 * min(f['hov'], 1.0)
         + 0.25 * (1.0 - min(f['dcx'] / 0.4, 1.0)) + 0.15 * (1.0 if 0.4 <= f['legs_in'] <= 1.15 else 0.0))
    return True, s, ''


def remove_people_in_vehicles(insts, review, cfg=None):
    """Bỏ người ngồi trong ô tô/xe buýt/xe tải (thấy qua kính): mask người nằm phần lớn trong mask xe,
    và đáy người cao hơn đáy xe (người đứng trước xe thì chân phải thấp hơn đáy xe)."""
    persons = [i for i in insts if i.cls == PED]
    vehs = [i for i in insts if i.cls in CLOSED_VEHICLES]
    drop = set()
    for p in persons:
        if p.src == 'human':
            continue
        for v in vehs:
            if box_ioa(p.box, v.box) < 0.5:
                continue
            ioa_mask = mask_inter(p, v) / max(1, p.area)
            vh = v.box[3] - v.box[1]
            if ioa_mask >= 0.5 and p.box[3] < v.box[3] - 0.05 * vh:
                drop.add(id(p))
                review.append(dict(reason='person_in_vehicle_dropped', cls=PED, box=p.box, conf=p.conf))
                break
    return [i for i in insts if id(i) not in drop]


def pair_riders(insts, cfg=None, review=None, refine_fn=None):
    """insts: list Inst đủ lớp. Trả về list Inst mới (người ngồi đã được gộp vào xe).

    refine_fn(boxes)->list[mask Inst|None]: tuỳ chọn, dùng SAM trên hộp hợp (người+xe) để lấy thêm hàng hoá/áo mưa.
    """
    cfg = cfg or {}
    review = review if review is not None else []
    persons = [i for i in insts if i.cls == PED]
    vehs = [i for i in insts if i.cls in RIDEABLE]
    if not persons or not vehs:
        return list(insts), review

    cands = []
    near = {}
    for pi, p in enumerate(persons):
        for vi, v in enumerate(vehs):
            if box_ioa(p.box, v.box) <= 0 and box_ioa(v.box, p.box) <= 0:
                continue
            f = features(p, v)
            r = max(2, int(0.03 * f['vh']))
            cpx = _contact(p, v, r)
            ok, s, why = is_riding(f, cpx, cfg)
            if v.cls == OTHER and ok:
                s *= 0.9  # ưu tiên xe 2 bánh hơn xích lô/ba gác khi mơ hồ
            if ok:
                cands.append((s, pi, vi))
            elif f['ioa_p'] > 0.10 and v.cls != OTHER:
                near.setdefault(pi, (vi, why))

    cands.sort(reverse=True)
    p_assigned, v_riders = {}, {}
    best_per_person = {}
    for s, pi, vi in cands:
        best_per_person.setdefault(pi, []).append((s, vi))
    for s, pi, vi in cands:
        if pi in p_assigned:
            continue
        v = vehs[vi]
        if len(v_riders.get(vi, [])) >= CAPACITY[v.cls]:
            continue
        p_assigned[pi] = (vi, s)
        v_riders.setdefault(vi, []).append(pi)

    out = [i for i in insts if i.cls not in RIDEABLE and i.cls != PED]
    used_p = set(p_assigned)
    merged_boxes, merged_insts = [], []
    for vi, v in enumerate(vehs):
        riders = v_riders.get(vi, [])
        if not riders:
            out.append(v)
            continue
        parts = [v] + [persons[pi] for pi in riders]
        new_cls = RIDEABLE[v.cls]
        conf = min(x.conf for x in parts)
        src = 'human' if all(x.src == 'human' for x in parts) else 'merged'
        m = union_insts(parts, cls=new_cls, conf=conf, src=src,
                        meta={'merged_from': [x.src for x in parts], 'n_riders': len(riders)})
        if m is None:
            out.append(v)
            continue
        scores = [p_assigned[pi][1] for pi in riders]
        for pi in riders:
            alts = [x for x in best_per_person.get(pi, []) if x[1] != vi]
            if alts and abs(alts[0][0] - p_assigned[pi][1]) < 0.08:
                review.append(dict(reason='pair_ambiguous_2_vehicles', cls=new_cls, box=m.box, conf=m.conf))
        if min(scores) < cfg.get('borderline', 0.55):
            review.append(dict(reason='pair_borderline', cls=new_cls, box=m.box, conf=m.conf, score=round(min(scores), 3)))
        if v.cls == OTHER:
            review.append(dict(reason='person_on_other_vehicle_merged', cls=OTHER, box=m.box, conf=m.conf))
        out.append(m)
        merged_boxes.append(m.box)
        merged_insts.append(m)

    for pi, p in enumerate(persons):
        if pi in used_p:
            continue
        out.append(p)
        if pi in near:
            review.append(dict(reason='person_near_two_wheeler_kept_as_pedestrian', cls=PED, box=p.box, conf=p.conf,
                               why=near[pi][1]))

    # tinh chỉnh: SAM trên hộp hợp để lấy cả hàng hoá buộc sau yên / tà áo mưa (chỉ chấp nhận nếu khớp)
    if refine_fn is not None and merged_boxes:
        refined = refine_fn(merged_boxes)
        for m, r in zip(merged_insts, refined):
            if r is None:
                continue
            inter = mask_inter(m, r)
            iou = inter / max(1, m.area + r.area - inter)
            grow = (r.area - inter) / max(1, m.area)
            if iou >= cfg.get('refine_min_iou', 0.6) and grow <= 0.35:
                nm = union_insts([m, r], cls=m.cls, conf=m.conf, src=m.src, meta=m.meta)
                if nm is not None:
                    m.mask, m.ox, m.oy, m.line = nm.mask, nm.ox, nm.oy, None
    return out, review
