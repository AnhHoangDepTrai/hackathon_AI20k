"""Lõi dùng chung cho toàn bộ tool của Thành viên 2.

Chỉ phụ thuộc numpy + opencv (chạy được trên Windows, Colab, Kaggle, không cần GPU).
- Bảng 11 lớp chuẩn (đúng thứ tự của hệ thống chấm)
- Đọc/ghi ảnh an toàn với đường dẫn có dấu/khoảng trắng trên Windows
- Đọc/ghi nhãn YOLO-seg
- Lớp Inst: một instance = mask cục bộ (cắt sát bbox) + toạ độ gốc
- Chuyển đổi polygon <-> mask, gộp nhiều mảnh thành 1 polygon
- Lọc vật quá nhỏ theo đúng ngưỡng của cuộc thi
- Giải quyết chồng lấn (Zero-Overlap) theo chiều sâu
- Perceptual hash để chống rò rỉ golden_val
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import sys

import cv2
import numpy as np

# Windows: console/pipe mặc định cp1258/cp1252 không in được tiếng Việt -> ép UTF-8, ký tự lạ thì thay thế (không crash)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# ----------------------------------------------------------------------------------------------- lớp
CLASSES = ['Car', 'Bus', 'Truck', 'Other vehicle', 'Motorcycle', 'Motorcyclist',
           'Bicycle', 'Cyclist', 'Pedestrian', 'Sign', 'Traffic light']
CAR, BUS, TRUCK, OTHER, MOTO, MOTORIST, BIKE, CYCLIST, PED, SIGN, LIGHT = range(11)
VEHICLES = {CAR, BUS, TRUCK, OTHER}
TWO_WHEEL = {MOTO, MOTORIST, BIKE, CYCLIST}
GROUND = {CAR, BUS, TRUCK, OTHER, MOTO, MOTORIST, BIKE, CYCLIST, PED}
ELEVATED = {SIGN, LIGHT}
GROUP = {CAR: 'veh', BUS: 'veh', TRUCK: 'veh', OTHER: 'veh', MOTO: 'tw', MOTORIST: 'tw', BIKE: 'tw',
         CYCLIST: 'tw', PED: 'ped', SIGN: 'sign', LIGHT: 'light'}
# Chiều cao thực tế xấp xỉ (mét) để ước lượng tỉ lệ khi Copy-Paste
PRIOR_HEIGHT_M = {CAR: 1.5, BUS: 3.2, TRUCK: 3.0, OTHER: 1.9, MOTO: 1.1, MOTORIST: 1.6,
                  BIKE: 1.0, CYCLIST: 1.7, PED: 1.65}
COLORS = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245), (180, 30, 145),
          (240, 240, 70), (230, 50, 240), (60, 245, 210), (212, 190, 250), (128, 128, 0)]  # BGR

# Tên lớp ở các bộ dữ liệu công khai -> id chuẩn. "4"/"6" sẽ được nâng lên 5/7 nếu có người ngồi (pairing).
NAME_TO_ID = {
    'car': CAR, 'sedan': CAR, 'suv': CAR, 'taxi': CAR, 'van': CAR, 'pickup': CAR, 'pickup truck': CAR,
    'minivan': CAR, 'ambulance': CAR, 'o to': CAR, 'oto': CAR,
    'bus': BUS, 'coach': BUS, 'minibus': BUS,
    'truck': TRUCK, 'lorry': TRUCK, 'fire truck': TRUCK, 'dump truck': TRUCK,
    'other vehicle': OTHER, 'other_vehicle': OTHER, 'tricycle': OTHER, 'cyclo': OTHER, 'trailer': OTHER,
    'container': OTHER, 'construction vehicle': OTHER, 'tractor': OTHER, 'xich lo': OTHER, 'ba gac': OTHER,
    'motorcycle': MOTO, 'motorbike': MOTO, 'motor': MOTO, 'scooter': MOTO, 'moto': MOTO, 'xe may': MOTO,
    'motorcyclist': MOTORIST,
    'bicycle': BIKE, 'bike': BIKE, 'xe dap': BIKE,
    'cyclist': CYCLIST,
    'person': PED, 'pedestrian': PED, 'people': PED, 'rider': PED, 'human': PED,
    'sign': SIGN, 'traffic sign': SIGN, 'traffic_sign': SIGN, 'road sign': SIGN, 'stop sign': SIGN,
    'traffic light': LIGHT, 'traffic_light': LIGHT, 'trafficlight': LIGHT,
}

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def norm_name(name):
    return re.sub(r'[\s_\-]+', ' ', str(name).strip().lower())


def name_to_id(name):
    """Tên lớp bất kỳ -> id chuẩn (None nếu không biết)."""
    n = norm_name(name)
    for i, c in enumerate(CLASSES):
        if n == norm_name(c):
            return i
    return NAME_TO_ID.get(n, NAME_TO_ID.get(n.replace(' ', '_')))


# ----------------------------------------------------------------------------------------------- IO ảnh
def imread(path):
    """cv2.imread không đọc được đường dẫn có dấu trên Windows -> dùng imdecode."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img, quality=95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or '.jpg'
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f'Không ghi được ảnh {path}')
    buf.tofile(str(path))


def list_images(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS and p.is_file()
                  and 'labels' not in [x.lower() for x in p.relative_to(root).parts[:-1]])


def label_rel_for(img_rel):
    """images/train/a.jpg -> labels/train/a.txt ; train/images/a.jpg -> train/labels/a.txt."""
    parts = list(Path(img_rel).parts)
    low = [p.lower() for p in parts]
    if 'images' in low[:-1]:
        i = len(low) - 2 - low[:-1][::-1].index('images')
        parts[i] = 'labels'
    else:
        parts.insert(0, 'labels')
    return Path(*parts).with_suffix('.txt')


def out_img_rel(img_rel):
    parts = [p.lower() for p in Path(img_rel).parts[:-1]]
    return Path(img_rel) if 'images' in parts else Path('images') / img_rel


def find_label(img_path, root):
    """Tìm tệp nhãn của ảnh: cấu trúc images/->labels/ hoặc .txt nằm cạnh ảnh."""
    img_path, root = Path(img_path), Path(root)
    cands = []
    try:
        cands.append(root / label_rel_for(img_path.relative_to(root)))
    except ValueError:
        pass
    cands.append(img_path.with_suffix('.txt'))
    parts = list(img_path.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == 'images':
            cands.append(Path(*parts[:i], 'labels', *parts[i + 1:]).with_suffix('.txt'))
            break
    for c in cands:
        if c.is_file():
            return c
    return None


# ----------------------------------------------------------------------------------------------- nhãn YOLO-seg
def read_label_file(path):
    """Trả về list (cls, pts[N,2] chuẩn hoá, kind). kind = 'poly' | 'box' (dòng 5 cột)."""
    out = []
    if path is None or not Path(path).is_file():
        return out
    for ln, line in enumerate(Path(path).read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
        p = line.split()
        if not p:
            continue
        try:
            c = int(float(p[0]))
            v = np.array([float(x) for x in p[1:]], dtype=np.float64)
        except ValueError:
            continue
        if len(v) == 4:  # bbox cx cy w h -> hình chữ nhật (đánh dấu để duyệt lại)
            cx, cy, w, h = v
            pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
            out.append((c, np.clip(pts, 0, 1), 'box'))
        elif len(v) >= 6 and len(v) % 2 == 0:
            out.append((c, np.clip(v.reshape(-1, 2), 0, 1), 'poly'))
    return out


def format_line(cls, pts_norm):
    pts = np.clip(np.asarray(pts_norm, dtype=np.float64), 0.0, 1.0)
    return str(int(cls)) + ' ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts)


# ----------------------------------------------------------------------------------------------- mask <-> polygon
def rasterize_local(pts_px, H, W):
    """Polygon (pixel) -> (mask cục bộ bool, ox, oy). Dùng shift=4 để giữ độ chính xác dưới pixel."""
    pts = np.asarray(pts_px, dtype=np.float64)
    x0 = int(max(0, math.floor(pts[:, 0].min())))
    y0 = int(max(0, math.floor(pts[:, 1].min())))
    x1 = int(min(W, math.ceil(pts[:, 0].max()) + 1))
    y1 = int(min(H, math.ceil(pts[:, 1].max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    q = np.round((pts - [x0, y0]) * 16).astype(np.int32)
    cv2.fillPoly(canvas, [q], 1, lineType=cv2.LINE_8, shift=4)
    return canvas.astype(bool), x0, y0


def tight(mask, ox, oy):
    """Cắt mask sát vùng có pixel. Trả về (mask, ox, oy) hoặc (None,0,0) nếu rỗng."""
    if mask is None or not mask.any():
        return None, 0, 0
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], ox + int(cols[0]), oy + int(rows[0])


def _bridge(res, s):
    """Nối đa giác s vào res bằng cầu độ rộng 0 tại cặp điểm gần nhất."""
    d = ((res[:, None, :] - s[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(d.argmin()), d.shape)
    return np.concatenate([res[i:], res[:i + 1], s[j:], s[:j + 1]])


def mask_to_polygon(mask, ox=0, oy=0, eps=None, min_comp_frac=0.04, holes=True):
    """Mask -> 1 polygon pixel (N,2), hoặc None.
    - nhiều mảnh rời (vật bị cột điện che) -> nối thành 1 polygon bằng cầu độ rộng 0
    - LỖ bên trong (vật khác đứng trước, nằm gọn trong vật này) -> giữ lỗ bằng đường cắt 'lỗ khoá',
      để Zero-Overlap vẫn đúng khi tô lại (cv2.fillPoly / pycocotools đều tô theo even-odd)
    - mảnh vụn / lỗ < min_comp_frac mảnh lớn nhất bị bỏ."""
    if mask is None or not mask.any():
        return None
    m = np.pad(mask.astype(np.uint8), 1)
    mode = cv2.RETR_CCOMP if holes else cv2.RETR_EXTERNAL
    found = cv2.findContours(m, mode, cv2.CHAIN_APPROX_NONE)
    contours, hier = (found[0], found[1]) if len(found) == 2 else (found[1], found[2])
    if not contours:
        return None
    hier = hier.reshape(-1, 4) if hier is not None else np.full((len(contours), 4), -1)
    areas = [max(cv2.contourArea(c), len(c) * 0.5) for c in contours]
    outer_ids = [k for k in range(len(contours)) if hier[k][3] < 0]
    amax = max(areas[k] for k in outer_ids)
    if eps is None:
        eps = 0.7 if max(mask.shape) < 400 else 1.0

    def simp(c):
        s = cv2.approxPolyDP(c, eps, True) if len(c) > 8 else c
        s = s.reshape(-1, 2).astype(np.float64)
        return s if len(s) >= 3 else c.reshape(-1, 2).astype(np.float64)

    segs = []
    for k in outer_ids:
        if areas[k] < max(3.0, min_comp_frac * amax):
            continue
        seg = simp(contours[k])
        if len(seg) < 3:
            continue
        if holes:
            ch = hier[k][2]
            while ch >= 0:
                if areas[ch] >= max(6.0, 0.002 * areas[k]):
                    hs = simp(contours[ch])
                    if len(hs) >= 3:
                        seg = _bridge(seg, hs)
                ch = hier[ch][0]
        segs.append(seg)
    if not segs:
        return None
    segs.sort(key=lambda s: -len(s))
    poly = segs[0]
    rest = segs[1:]
    while rest:
        best = None
        for k, sg in enumerate(rest):
            d = ((poly[:, None, :] - sg[None, :, :]) ** 2).sum(-1).min()
            if best is None or d < best[0]:
                best = (d, k)
        poly = _bridge(poly, rest.pop(best[1]))
    return poly + [ox - 1, oy - 1]


# ----------------------------------------------------------------------------------------------- Instance
@dataclass(eq=False)  # so sánh theo đối tượng (không so mảng numpy)
class Inst:
    cls: int
    mask: np.ndarray          # bool, cục bộ, đã cắt sát
    ox: int
    oy: int
    conf: float = 1.0
    src: str = 'human'        # human | det | ov | merged | paste
    line: str | None = None   # dòng nhãn gốc: giữ nguyên nếu mask không đổi (không làm xấu polygon người vẽ)
    meta: dict = field(default_factory=dict)

    @property
    def box(self):
        h, w = self.mask.shape
        return (self.ox, self.oy, self.ox + w, self.oy + h)

    @property
    def area(self):
        return int(self.mask.sum())

    def full(self, H, W):
        m = np.zeros((H, W), bool)
        x0, y0, x1, y1 = self.box
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = self.mask[ya - y0:yb - y0, xa - x0:xb - x0]
        return m

    def crop(self, x0, y0, x1, y1):
        """Phần mask nằm trong khung [x0,x1)x[y0,y1) (toạ độ ảnh) dạng mảng (y1-y0, x1-x0)."""
        out = np.zeros((y1 - y0, x1 - x0), bool)
        bx0, by0, bx1, by1 = self.box
        xa, ya, xb, yb = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
        if xb > xa and yb > ya:
            out[ya - y0:yb - y0, xa - x0:xb - x0] = self.mask[ya - by0:yb - by0, xa - bx0:xb - bx0]
        return out

    @staticmethod
    def from_mask(mask, ox=0, oy=0, **kw):
        m, ox, oy = tight(np.asarray(mask, bool), ox, oy)
        return None if m is None else Inst(mask=m, ox=ox, oy=oy, **kw)

    @staticmethod
    def from_polygon(pts_px, H, W, **kw):
        m, ox, oy = rasterize_local(pts_px, H, W)
        return Inst.from_mask(m, ox, oy, **kw) if m is not None else None

    def polygon_px(self):
        return mask_to_polygon(self.mask, self.ox, self.oy)

    def to_line(self, H, W):
        if self.line is not None:
            return self.line
        poly = self.polygon_px()
        if poly is None or len(poly) < 3:
            return None
        return format_line(self.cls, poly / [W, H])


def union_insts(insts, **kw):
    x0 = min(i.box[0] for i in insts)
    y0 = min(i.box[1] for i in insts)
    x1 = max(i.box[2] for i in insts)
    y1 = max(i.box[3] for i in insts)
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for i in insts:
        m |= i.crop(x0, y0, x1, y1)
    return Inst.from_mask(m, x0, y0, **kw)


def mask_inter(a: Inst, b: Inst):
    x0, y0 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
    x1, y1 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((a.crop(x0, y0, x1, y1) & b.crop(x0, y0, x1, y1)).sum())


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_ioa(a, b):
    """Diện tích giao / diện tích của a."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / aa if aa > 0 else 0.0


# ----------------------------------------------------------------------------------------------- lọc
def too_small(inst: Inst, H, W, min_area640=64.0, min_short_frac=0.0125):
    """Đúng ngưỡng cuộc thi: mask < 64 px ở thang 640 bị bỏ qua khi chấm; cạnh ngắn < 1.25% cạnh dài ảnh."""
    long_side = max(H, W)
    x0, y0, x1, y1 = inst.box
    if min(x1 - x0, y1 - y0) < min_short_frac * long_side:
        return True
    return inst.area * (640.0 / long_side) ** 2 < min_area640


def content_box(img, thr=10):
    """Vùng có nội dung thật (bỏ viền đen letterbox của Roboflow)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    rows = np.flatnonzero(g.max(1) > thr)
    cols = np.flatnonzero(g.max(0) > thr)
    H, W = g.shape
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, W, H
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def is_ego_hood(inst: Inst, cbox):
    """Nắp capo/taplo của chính xe gắn dashcam: chạm đáy khung hình, rất rộng, nằm thấp."""
    cx0, cy0, cx1, cy1 = cbox
    x0, y0, x1, y1 = inst.box
    cw, ch = cx1 - cx0, cy1 - cy0
    return (y1 >= cy1 - 3) and (x1 - x0) >= 0.5 * cw and y0 >= cy0 + 0.55 * ch and inst.cls in (CAR, TRUCK, BUS, OTHER)


def sanitize_inst(inst: Inst, k=3, min_shift=3):
    """Xoá 'gai'/đường kẻ mảnh do polygon lỗi (đỉnh bay xa, xem ảnh Roboflow có đường chéo dài).
    Mở hình thái học để bỏ phần mảnh < k px, rồi trả lại các góc bị bào (giao với bản nới của phần thân).
    Chỉ chấp nhận khi bbox thật sự co lại >= min_shift px (tức là có gai), để không làm xấu vật nhỏ hợp lệ.
    Trả về (inst_mới, có_đổi_không)."""
    if min(inst.mask.shape) < k + 3:
        return inst, False
    m = inst.mask.astype(np.uint8)
    ker = np.ones((k, k), np.uint8)
    op = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    if op.sum() < 0.5 * m.sum():
        return inst, False  # vật mảnh hợp lệ (cột đèn nhỏ, xe đạp xa): không đụng vào
    body = (m.astype(bool) & cv2.dilate(op, ker).astype(bool))
    nb, oxn, oyn = tight(body, inst.ox, inst.oy)
    if nb is None:
        return inst, False
    x0, y0, x1, y1 = inst.box
    bx0, by0 = oxn, oyn
    bx1, by1 = oxn + nb.shape[1], oyn + nb.shape[0]
    if max(bx0 - x0, by0 - y0, x1 - bx1, y1 - by1) < min_shift:
        return inst, False
    return Inst(cls=inst.cls, mask=nb, ox=oxn, oy=oyn, conf=inst.conf, src=inst.src, meta=dict(inst.meta)), True


def resolve_overlaps(insts, H, W, keep_tol=0.01):
    """Zero-Overlap: vật phía trước giữ pixel. Phía trước = đáy bbox thấp hơn (gần camera hơn).
    Biển báo / đèn luôn được vẽ trên cùng (thường đứng trước nền). Trả về list (inst_visible, lost_frac)."""
    if not insts:
        return []
    order = sorted(range(len(insts)), key=lambda k: (insts[k].cls in ELEVATED, insts[k].box[3], -insts[k].area))
    idmap = np.full((H, W), -1, np.int32)
    for k in order:
        x0, y0, x1, y1 = insts[k].box
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if xb <= xa or yb <= ya:
            continue
        sub = idmap[ya:yb, xa:xb]
        sub[insts[k].mask[ya - y0:yb - y0, xa - x0:xb - x0]] = k
    out = []
    for k, inst in enumerate(insts):
        x0, y0, x1, y1 = inst.box
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        a0 = inst.area
        if xb <= xa or yb <= ya or a0 == 0:
            out.append((None, 1.0))
            continue
        vis = (idmap[ya:yb, xa:xb] == k)
        a1 = int(vis.sum())
        lost = 1.0 - a1 / a0
        if lost <= 0:
            out.append((inst, 0.0))
            continue
        new = Inst.from_mask(vis, xa, ya, cls=inst.cls, conf=inst.conf, src=inst.src, meta=dict(inst.meta))
        if new is not None and inst.line is not None and lost <= keep_tol:
            new.line = inst.line  # chênh lệch do răng cưa biên: giữ polygon gốc của người vẽ
        out.append((new, lost))
    return out


# ----------------------------------------------------------------------------------------------- perceptual hash
def phash(img, content=True):
    if content:
        x0, y0, x1, y1 = content_box(img)
        if (x1 - x0) > 16 and (y1 - y0) > 16:
            img = img[y0:y1, x0:x1]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(s)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int(''.join('1' if b else '0' for b in bits), 2)


class HashIndex:
    def __init__(self):
        self.hashes = np.zeros(0, np.uint64)
        self.names = []

    def add(self, h, name):
        self.hashes = np.append(self.hashes, np.uint64(h))
        self.names.append(name)

    def nearest(self, h):
        if len(self.hashes) == 0:
            return 99, None
        x = np.bitwise_xor(self.hashes, np.uint64(h))
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        i = int(d.argmin())
        return int(d[i]), self.names[i]

    @staticmethod
    def from_dir(root, log=print):
        idx = HashIndex()
        if not root:
            return idx
        for p in list_images(root):
            img = imread(p)
            if img is not None:
                idx.add(phash(img), str(p))
        log(f'[golden_val] đã băm {len(idx.names)} ảnh từ {root}')
        return idx


# ----------------------------------------------------------------------------------------------- tiện ích
def write_data_yaml(out_dir, splits=None):
    out_dir = Path(out_dir)
    lines = ['# Sinh tự động bởi tools_tv2 — 11 lớp chuẩn cuộc thi', 'path: .']
    if splits:
        for k, v in splits.items():
            lines.append(f'{k}: {v}')
    else:
        lines += ['train: images', 'val: images']
    lines.append(f'nc: {len(CLASSES)}')
    lines.append('names: [' + ', '.join(f"'{c}'" for c in CLASSES) + ']')
    (out_dir / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def class_counts_from_dir(lbl_root):
    cnt = [0] * len(CLASSES)
    for f in Path(lbl_root).rglob('*.txt'):
        for c, _, _ in read_label_file(f):
            if 0 <= c < len(CLASSES):
                cnt[c] += 1
    return {CLASSES[i]: cnt[i] for i in range(len(CLASSES))}


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def parse_int_list(s):
    if s is None or str(s).strip().lower() in ('', 'none'):
        return []
    if str(s).strip().lower() == 'all':
        return list(range(len(CLASSES)))
    return [int(x) for x in str(s).replace(' ', '').split(',') if x != '']


def progress(i, n, t0, every=25, prefix=''):
    if i == n or i % every == 0:
        el = time.time() - t0
        eta = el / max(i, 1) * (n - i)
        print(f'{prefix}{i}/{n} ảnh | đã chạy {el/60:.1f} phút | còn ~{eta/60:.1f} phút', flush=True)


def load_src_map(spec, yaml_path=None, log=print):
    """Bảng ánh xạ id lớp gốc -> id chuẩn.
    spec: 'auto' (đọc names trong data.yaml), 'identity' (đã là 11 lớp chuẩn), hoặc '0:6,1:1,2:0' (gốc:chuẩn, -1 = bỏ)."""
    if spec in (None, '', 'identity'):
        return {i: i for i in range(len(CLASSES))}, 'identity'
    if spec != 'auto':
        m = {}
        for kv in str(spec).split(','):
            if kv.strip():
                a, b = kv.split(':')
                m[int(a)] = int(b)
        return m, 'manual'
    if yaml_path is None or not Path(yaml_path).is_file():
        log('[map] không tìm thấy data.yaml -> coi như nhãn đã là 11 lớp chuẩn')
        return {i: i for i in range(len(CLASSES))}, 'identity'
    names = _read_yaml_names(yaml_path)
    m = {}
    for i, n in enumerate(names):
        t = name_to_id(n)
        m[i] = -1 if t is None else t
        log(f'[map] gốc {i} "{n}" -> ' + (f'{t} {CLASSES[t]}' if t is not None else 'BỎ (không biết lớp này)'))
    return m, 'auto:' + str(yaml_path)


def _read_yaml_names(path):
    try:
        import yaml  # noqa: PLC0415
        d = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        names = d.get('names')
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        return list(names)
    except ImportError:
        txt = Path(path).read_text(encoding='utf-8')
        mm = re.search(r'names:\s*\[(.*?)\]', txt, re.S)
        return [x.strip().strip('\'"') for x in mm.group(1).split(',')] if mm else []


def find_data_yaml(root):
    root = Path(root)
    for p in [root, *root.parents][:3]:
        if (p / 'data.yaml').is_file():
            return p / 'data.yaml'
    return None


def load_insts(img_path, root, H, W, src_map, src='human', log=None):
    """Đọc nhãn có sẵn của ảnh thành list Inst (đã ánh xạ lớp). Trả về (insts, n_box_lines)."""
    lf = find_label(img_path, root)
    insts, nbox = [], 0
    for c, pts, kind in read_label_file(lf):
        t = src_map.get(c, -1)
        if t is None or t < 0 or t >= len(CLASSES):
            continue
        it = Inst.from_polygon(pts * [W, H], H, W, cls=t, src=src)
        if it is None:
            continue
        if kind == 'box':
            nbox += 1
            it.meta['was_box'] = True
        else:
            it.line = format_line(t, pts)  # giữ nguyên polygon gốc (chỉ đổi id lớp) nếu không bị sửa
        insts.append(it)
    return insts, nbox


def relpath_str(p, root):
    try:
        return str(Path(p).relative_to(root)).replace(os.sep, '/')
    except ValueError:
        return Path(p).name
