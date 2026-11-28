"""Kiểm thử toàn luồng autolabel.py với model GIẢ (không cần GPU):  python tests/test_pipeline_mock.py"""
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import autolabel  # noqa: E402
from tv2lib.core import Inst, read_label_file  # noqa: E402


class FakeModels:
    """Giả lập detector+SAM: trả về 1 người ngồi trên xe máy, 1 người đi bộ, 1 biển báo, 1 'container' trên xe tải."""
    def __init__(self, args, log=print):
        self.args = args

    def run(self, img, review):
        H, W = img.shape[:2]
        def r(x0, y0, x1, y1, c, conf=0.9, src='det'):
            m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
            return Inst.from_mask(m, cls=c, conf=conf, src=src, meta={'name': 'fake'})
        rider = r(285, 200, 345, 330, 8)
        walker = r(60, 220, 90, 380, 8, 0.45)
        sign = r(500, 60, 530, 90, 9, 0.8, 'ov')
        cont = r(430, 180, 560, 300, 3, 0.7, 'ov')
        return [rider, walker, sign], [cont]

    def refine_fn(self, img):
        return None


def make_dataset(d):
    (d / 'train' / 'images').mkdir(parents=True)
    (d / 'train' / 'labels').mkdir(parents=True)
    (d / 'data.yaml').write_text("names: ['bicycle', 'bus', 'car', 'motorcycle', 'truck']\nnc: 5\n")
    img = np.full((480, 640, 3), 90, np.uint8)
    img[:60] = 0  # viền đen letterbox
    img[420:] = 0
    cv2.imwrite(str(d / 'train' / 'images' / 'a.jpg'), img)
    W, H = 640, 480
    def poly(c, x0, y0, x1, y1):
        return f'{c} {x0/W} {y0/H} {x1/W} {y0/H} {x1/W} {y1/H} {x0/W} {y1/H}'
    lines = [poly(3, 290, 280, 340, 360),      # motorcycle (chỉ xe) -> phải thành Motorcyclist
             poly(2, 100, 250, 250, 350),      # car -> 0
             poly(4, 400, 170, 600, 320),      # truck có container -> tách
             poly(2, 40, 380, 600, 420)]       # nắp capo dashcam -> bỏ
    (d / 'train' / 'labels' / 'a.txt').write_text('\n'.join(lines) + '\n')


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        make_dataset(tmp / 'rf')
        autolabel.Models = FakeModels
        args = autolabel.build_parser().parse_args(['--images', str(tmp / 'rf'), '--out', str(tmp / 'out'),
                                                    '--keep-labels', '--split-trailer', '--viz', '2'])
        autolabel.run(args)
        items = read_label_file(tmp / 'out' / 'train' / 'labels' / 'a.txt')
        cls = sorted(c for c, _, _ in items)
        print('Lớp đầu ra:', cls)
        assert cls == [0, 2, 3, 5, 8, 9], cls  # car, truck(cabin), container, motorcyclist, pedestrian, sign
        rev = (tmp / 'out' / 'review.csv').read_text(encoding='utf-8-sig')
        assert 'ego_hood_dropped' in rev and 'trailer_split_from_truck' in rev and 'low_conf_auto_label' in rev
        assert (tmp / 'out' / 'data.yaml').is_file() and (tmp / 'out' / 'run_log.json').is_file()
        assert any((tmp / 'out' / 'viz').iterdir())
        print('\nPIPELINE MOCK ĐẠT')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
