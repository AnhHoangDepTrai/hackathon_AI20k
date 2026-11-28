"""Chạy autolabel.py với module 'ultralytics' GIẢ để kiểm tra phần nối model (không cần GPU/mạng).
python tests/test_fake_ultralytics.py"""
import shutil
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class T:  # giả torch tensor
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def int(self):
        return T(self.a.astype(int))

    def __len__(self):
        return len(self.a)


class Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = T(np.array(xyxy, float).reshape(-1, 4)), T(np.array(cls, float)), T(np.array(conf, float))

    def __len__(self):
        return len(self.xyxy)


class Masks:
    def __init__(self, data):
        self.data = T(data)


class R:
    def __init__(self, boxes=None, masks=None):
        self.boxes, self.masks = boxes, masks


COCO = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'}


class YOLO:
    def __init__(self, w):
        self.names = dict(COCO)

    def predict(self, img, **kw):
        # 1 xe máy + 1 người ngồi trên + 1 đèn giao thông
        return [R(Boxes([[300, 280, 360, 380], [290, 200, 370, 350], [500, 40, 515, 80]], [3, 0, 9], [0.9, 0.85, 0.6]))]


class YOLOWorld:
    def __init__(self, w):
        self.cls = []

    def set_classes(self, names):
        self.names = names

    def predict(self, img, **kw):
        i = self.names.index('traffic sign')
        return [R(Boxes([[560, 50, 590, 80]], [i], [0.5]))]


class SAM:
    def __init__(self, w):
        pass

    def predict(self, img, bboxes=None, **kw):
        H, W = img.shape[:2]
        out = np.zeros((len(bboxes), H, W), np.float32)
        for k, b in enumerate(bboxes):
            x0, y0, x1, y1 = map(int, b)
            cv2.ellipse(out[k], ((x0 + x1) // 2, (y0 + y1) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1, -1)
        return [R(masks=Masks(out))]


def main():
    fake = types.ModuleType('ultralytics')
    fake.YOLO, fake.YOLOWorld, fake.SAM = YOLO, YOLOWorld, SAM
    sys.modules['ultralytics'] = fake
    import autolabel  # noqa: PLC0415
    from tv2lib.core import read_label_file  # noqa: PLC0415
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / 'imgs').mkdir()
        cv2.imwrite(str(tmp / 'imgs' / 'x.jpg'), np.full((480, 640, 3), 120, np.uint8))
        args = autolabel.build_parser().parse_args(['--images', str(tmp / 'imgs'), '--out', str(tmp / 'out'), '--device', 'cpu'])
        autolabel.run(args)
        items = read_label_file(tmp / 'out' / 'labels' / 'x.txt')
        cls = sorted(c for c, _, _ in items)
        print('Lớp:', cls)
        assert cls == [5, 9, 10], cls
        print('\nFAKE-ULTRALYTICS ĐẠT')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
