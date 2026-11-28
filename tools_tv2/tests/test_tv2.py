"""Kiểm thử nhanh (không cần GPU/model):  python tests/test_tv2.py"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tv2lib.core import (BIKE, CAR, CYCLIST, MOTO, MOTORIST, PED, Inst, format_line, mask_to_polygon,  # noqa: E402
                         rasterize_local, read_label_file, resolve_overlaps, sanitize_inst, too_small, is_ego_hood)
from tv2lib.pairing import pair_riders, remove_people_in_vehicles  # noqa: E402

H, W = 720, 1280


def ellipse(cx, cy, ax, ay, cls, src='det'):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    return Inst.from_mask(m.astype(bool), cls=cls, src=src)


def rect(x0, y0, x1, y1, cls, src='det'):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return Inst.from_mask(m, cls=cls, src=src)


def iou(a, b):
    A, B = a.full(H, W), b.full(H, W)
    return (A & B).sum() / max(1, (A | B).sum())


def test_roundtrip():
    e = ellipse(400, 300, 80, 50, CAR)
    poly = mask_to_polygon(e.mask, e.ox, e.oy)
    back = Inst.from_polygon(poly, H, W, cls=CAR)
    assert iou(e, back) > 0.97, iou(e, back)
    # 2 mảnh rời (vật bị cột điện che) -> 1 polygon duy nhất vẫn phủ cả 2 mảnh
    m = np.zeros((H, W), bool)
    m[100:200, 100:180] = True
    m[100:200, 200:280] = True
    two = Inst.from_mask(m, cls=CAR)
    p2 = mask_to_polygon(two.mask, two.ox, two.oy)
    b2 = Inst.from_polygon(p2, H, W, cls=CAR)
    assert iou(two, b2) > 0.95, iou(two, b2)
    # ghi/đọc dòng nhãn
    ln = two.to_line(H, W)
    assert ln.startswith('0 ') and len(ln.split()) >= 7
    print('roundtrip OK  (ellipse IoU %.3f, 2-mảnh IoU %.3f)' % (iou(e, back), iou(two, b2)))


def test_pairing():
    # xe máy nhìn từ sau, người ngồi trên: hộp người phủ nửa trên xe
    bike = rect(600, 420, 660, 560, MOTO, 'human')
    rider = rect(585, 300, 675, 500, PED)
    # xe máy đỗ, người đứng cạnh (dắt xe)
    bike2 = rect(900, 450, 1080, 560, MOTO, 'human')
    walker = rect(1070, 330, 1130, 565, PED)
    # xe đạp chở 2 người
    cyc = rect(200, 430, 330, 540, BIKE)
    r1 = rect(230, 330, 290, 500, PED)
    r2 = rect(260, 340, 320, 505, PED)
    # người đi bộ xa mọi xe
    ped = rect(50, 300, 90, 420, PED)
    review = []
    out, review = pair_riders([bike, rider, bike2, walker, cyc, r1, r2, ped], {}, review)
    cls = sorted(i.cls for i in out)
    assert cls.count(MOTORIST) == 1 and cls.count(MOTO) == 1, cls
    assert cls.count(CYCLIST) == 1 and cls.count(PED) == 2, cls
    mr = [i for i in out if i.cls == MOTORIST][0]
    assert mr.box == (585, 300, 675, 560), mr.box     # gộp người + xe = 1 instance
    cy = [i for i in out if i.cls == CYCLIST][0]
    assert cy.meta['n_riders'] == 2
    print('pairing OK   ->', [i.cls for i in out], '| review:', [r['reason'] for r in review])


def test_side_view_walker():
    # nhìn ngang: người dắt xe đứng ở đầu xe
    bike = rect(400, 450, 600, 560, MOTO, 'human')
    walker = rect(570, 320, 640, 562, PED)
    # nhìn ngang: người ngồi giữa xe
    bike2 = rect(800, 450, 1000, 560, MOTO, 'human')
    rider = rect(850, 330, 950, 520, PED)
    out, rv = pair_riders([bike, walker, bike2, rider])
    assert sorted(i.cls for i in out) == sorted([MOTO, PED, MOTORIST]), [i.cls for i in out]
    print('side-view OK ->', [i.cls for i in out])


def test_person_in_car():
    car = rect(300, 300, 700, 560, CAR, 'human')
    inside = rect(450, 330, 520, 420, PED)          # thấy qua kính
    front = rect(470, 380, 530, 640, PED)           # đứng trước xe: chân thấp hơn đáy xe
    out = remove_people_in_vehicles([car, inside, front], [])
    assert len(out) == 2 and inside not in out and front in out
    print('person-in-car OK')


def test_overlap_and_filters():
    back = rect(100, 100, 400, 300, CAR, 'human')
    front = rect(300, 200, 500, 400, CAR)
    res = resolve_overlaps([back, front], H, W)
    vb, lb = res[0]
    vf, lf = res[1]
    assert lf == 0 and vb.area == back.area - 100 * 100, (vb.area, back.area)
    assert not (vb.full(H, W) & vf.full(H, W)).any()
    tiny = rect(10, 10, 16, 40, PED)            # cạnh ngắn 6px < 1.25% * 1280 = 16px
    assert too_small(tiny, H, W)
    hood = rect(100, 600, 1180, 720, CAR, 'human')
    assert is_ego_hood(hood, (0, 0, W, H))
    # polygon có gai (đỉnh bay xa) -> sanitize xoá gai
    pts = np.array([[100, 100], [300, 100], [300, 250], [900, 700], [301, 252], [100, 250]], float)
    spike = Inst.from_polygon(pts, H, W, cls=CAR, src='human')
    clean, changed = sanitize_inst(spike)
    assert changed and clean.box[2] <= 305 and clean.box[3] <= 260, clean.box
    print('overlap/filter/sanitize OK')


if __name__ == '__main__':
    test_roundtrip()
    test_pairing()
    test_side_view_walker()
    test_person_in_car()
    test_overlap_and_filters()
    print('\nTẤT CẢ KIỂM THỬ ĐẠT')
