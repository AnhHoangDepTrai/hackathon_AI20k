"""Bộ phát hiện + SAM 2 (chỉ nạp torch/ultralytics khi thật sự dùng; cần GPU để chạy nhanh).

- UltraDetector : bất kỳ trọng số Ultralytics (yolo11x.pt, yolov8x.pt, rtdetr-x.pt, yolo11x-seg.pt...)
                  hoặc chính best.pt 11 lớp của đội (active learning). Ánh xạ lớp theo TÊN.
- WorldDetector : YOLO-World (open-vocabulary, có sẵn trong Ultralytics) cho lớp COCO không có:
                  biển báo, xích lô/ba gác/xe công trình, container/rơ-moóc.
- GDinoDetector : GroundingDINO (HuggingFace transformers) — chính xác hơn với biển báo, chậm hơn.
- SamSegmenter  : SAM 2 / 2.1 của Ultralytics, prompt bằng bbox -> mask từng vật.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .core import (CLASSES, LIGHT, OTHER, PED, SIGN, BIKE, MOTO, CAR, BUS, TRUCK, Inst, box_iou, name_to_id)

# conf tối thiểu theo lớp đích (detector COCO). Ưu tiên chính xác: nhãn nháp sai còn tệ hơn thiếu.
DEFAULT_CONF = {CAR: 0.40, BUS: 0.40, TRUCK: 0.40, OTHER: 0.35, MOTO: 0.30, 5: 0.35, BIKE: 0.30, 7: 0.35,
                PED: 0.35, SIGN: 0.35, LIGHT: 0.30}

# prompt open-vocab -> (lớp đích, loại)
OV_PROMPTS = {
    'traffic sign': (SIGN, 'sign'),
    'road sign': (SIGN, 'sign'),
    'traffic light': (LIGHT, 'light'),
    'cyclo': (OTHER, 'other'),
    'rickshaw': (OTHER, 'other'),
    'three-wheeled cargo cart': (OTHER, 'other'),
    'tricycle': (OTHER, 'other'),
    'excavator': (OTHER, 'other'),
    'road roller': (OTHER, 'other'),
    'tractor': (OTHER, 'other'),
    'shipping container': (OTHER, 'trailer'),
    'semi-trailer': (OTHER, 'trailer'),
}
OV_CONF = {'sign': 0.25, 'light': 0.25, 'other': 0.30, 'trailer': 0.30}


@dataclass
class Det:
    box: np.ndarray           # x1 y1 x2 y2 (pixel ảnh gốc)
    cls: int
    conf: float
    name: str
    src: str = 'det'          # det | ov
    kind: str = ''            # với ov: sign/light/other/trailer
    poly: np.ndarray | None = None  # polygon từ model seg (nếu có)
    meta: dict = field(default_factory=dict)


def _device(dev):
    if dev not in (None, '', 'auto'):
        return dev
    try:
        import torch  # noqa: PLC0415
        return 0 if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def nms_dets(dets, iou_thr=0.6, same_group=True):
    """NMS tham lam giữa các nguồn (COCO + open-vocab)."""
    from .core import GROUP  # noqa: PLC0415
    dets = sorted(dets, key=lambda d: -d.conf)
    keep = []
    for d in dets:
        dup = False
        for k in keep:
            if same_group and GROUP[k.cls] != GROUP[d.cls]:
                continue
            if box_iou(d.box, k.box) >= iou_thr:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


class UltraDetector:
    def __init__(self, weights='yolo11x.pt', imgsz=1280, device=None, class_conf=None, log=print):
        from ultralytics import YOLO  # noqa: PLC0415
        self.m = YOLO(weights)
        self.imgsz = imgsz
        self.device = _device(device)
        self.conf = dict(DEFAULT_CONF)
        if class_conf:
            self.conf.update(class_conf)
        names = self.m.names if isinstance(self.m.names, dict) else dict(enumerate(self.m.names))
        self.map = {}
        for i, n in names.items():
            t = name_to_id(n)
            if t is not None:
                self.map[int(i)] = t
        log(f'[det] {weights}: dùng {len(self.map)} lớp -> ' +
            ', '.join(f'{names[i]}->{CLASSES[t]}' for i, t in sorted(self.map.items())))
        if not self.map:
            raise RuntimeError(f'Model {weights} không có lớp nào ánh xạ được về 11 lớp chuẩn')

    def __call__(self, img):
        H, W = img.shape[:2]
        r = self.m.predict(img, imgsz=self.imgsz, conf=min(self.conf.values()) * 0.9, iou=0.6,
                           classes=sorted(self.map), max_det=300, verbose=False, device=self.device)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        polys = None
        if getattr(r, 'masks', None) is not None:
            try:
                polys = r.masks.xy  # polygon theo pixel ẢNH GỐC
            except Exception:  # noqa: BLE001
                polys = None
        for k in range(len(xyxy)):
            t = self.map.get(int(cls[k]))
            if t is None or conf[k] < self.conf.get(t, 0.35):
                continue
            b = np.clip(xyxy[k], [0, 0, 0, 0], [W, H, W, H])
            p = polys[k] if polys is not None and len(polys[k]) >= 3 else None
            out.append(Det(b, t, float(conf[k]), self.m.names[int(cls[k])], 'det', poly=p))
        return out


class WorldDetector:
    def __init__(self, weights='yolov8x-worldv2.pt', imgsz=1280, device=None, prompts=None, log=print):
        try:
            from ultralytics import YOLOWorld  # noqa: PLC0415
            self.m = YOLOWorld(weights)
        except Exception:  # noqa: BLE001
            from ultralytics import YOLO  # noqa: PLC0415
            self.m = YOLO(weights)
        self.prompts = dict(prompts or OV_PROMPTS)
        self.names = list(self.prompts)
        self.m.set_classes(self.names)
        self.imgsz = imgsz
        self.device = _device(device)
        log(f'[ov] YOLO-World {weights} với {len(self.names)} prompt: {", ".join(self.names)}')

    def __call__(self, img):
        H, W = img.shape[:2]
        r = self.m.predict(img, imgsz=self.imgsz, conf=min(OV_CONF.values()) * 0.9, iou=0.5, max_det=200,
                           verbose=False, device=self.device)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            name = self.names[c]
            t, kind = self.prompts[name]
            if s < OV_CONF[kind]:
                continue
            out.append(Det(np.clip(b, 0, [W, H, W, H]), t, float(s), name, 'ov', kind))
        return out


class GDinoDetector:
    def __init__(self, model_id='IDEA-Research/grounding-dino-base', device=None, prompts=None,
                 box_thr=0.30, text_thr=0.25, log=print):
        import torch  # noqa: PLC0415
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415
        self.torch = torch
        dev = _device(device)
        self.dev = f'cuda:{dev}' if isinstance(dev, int) else ('cuda' if str(dev).isdigit() else str(dev))
        if self.dev.startswith('cuda') and not torch.cuda.is_available():
            self.dev = 'cpu'
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.dev).eval()
        self.prompts = dict(prompts or OV_PROMPTS)
        self.text = '. '.join(self.prompts) + '.'
        self.box_thr, self.text_thr = box_thr, text_thr
        log(f'[ov] GroundingDINO {model_id} trên {self.dev}')

    def _match(self, label):
        label = str(label).lower().strip()
        best, bl = None, 0
        for p in self.prompts:
            if p in label or label in p:
                if len(p) > bl:
                    best, bl = p, len(p)
        return best

    def __call__(self, img):
        from PIL import Image  # noqa: PLC0415
        H, W = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inputs = self.proc(images=pil, text=self.text, return_tensors='pt').to(self.dev)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        pp = self.proc.post_process_grounded_object_detection
        try:
            res = pp(outputs, inputs.input_ids, threshold=self.box_thr, text_threshold=self.text_thr, target_sizes=[(H, W)])[0]
        except TypeError:
            res = pp(outputs, inputs.input_ids, box_threshold=self.box_thr, text_threshold=self.text_thr, target_sizes=[(H, W)])[0]
        labels = res.get('text_labels', res.get('labels'))
        out = []
        for b, s, lab in zip(res['boxes'].cpu().numpy(), res['scores'].cpu().numpy(), labels):
            p = self._match(lab)
            if p is None:
                continue
            t, kind = self.prompts[p]
            if s < OV_CONF[kind]:
                continue
            out.append(Det(np.clip(b, 0, [W, H, W, H]), t, float(s), p, 'ov', kind))
        return out


class SamSegmenter:
    def __init__(self, weights='sam2.1_b.pt', device=None, chunk=48, log=print):
        from ultralytics import SAM  # noqa: PLC0415
        self.m = SAM(weights)
        self.device = _device(device)
        self.chunk = chunk
        log(f'[sam] {weights}')

    def __call__(self, img, boxes, pad=0.10):
        """boxes: list x1y1x2y2 -> list Inst|None (mask bị cắt trong hộp nới pad để tránh SAM 'tràn' ra đường)."""
        H, W = img.shape[:2]
        out = []
        boxes = [list(map(float, b)) for b in boxes]
        for s in range(0, len(boxes), self.chunk):
            chunk = boxes[s:s + self.chunk]
            r = self.m.predict(img, bboxes=chunk, verbose=False, device=self.device)[0]
            md = None if r.masks is None else r.masks.data.cpu().numpy()
            for k, b in enumerate(chunk):
                if md is None or k >= len(md):
                    out.append(None)
                    continue
                m = md[k] > 0.5
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                bw, bh = b[2] - b[0], b[3] - b[1]
                x0 = int(max(0, b[0] - pad * bw)); y0 = int(max(0, b[1] - pad * bh))
                x1 = int(min(W, b[2] + pad * bw + 1)); y1 = int(min(H, b[3] + pad * bh + 1))
                out.append(Inst.from_mask(m[y0:y1, x0:x1], x0, y0, cls=0))
        return out
