#!/usr/bin/env python3
"""[TV2] Ghép cặp Người + Xe 2 bánh trên bộ nhãn CÓ SẴN (không cần GPU, không chạy model).

Dùng khi bộ dữ liệu đã có nhãn người và xe TÁCH RỜI (COCO, BDD100K, Cityscapes, OpenImages... sau khi remap),
hoặc khi người trong nhóm vẽ tay xe và người thành 2 polygon:
  - người NGỒI trên xe máy  -> gộp thành 1 polygon lớp 5 Motorcyclist
  - người NGỒI trên xe đạp  -> gộp thành 1 polygon lớp 7 Cyclist
  - người DẮT/đứng cạnh xe  -> giữ người = 8 Pedestrian, xe = 4 Motorcycle / 6 Bicycle
  - đồng thời: remap lớp, xoá gai polygon, bỏ nắp capo dashcam, xử lý chồng lấn (Zero-Overlap), lọc vật nhỏ.

Ví dụ:
  # dataset đã là 11 lớp chuẩn (người = 8, xe máy = 4, xe đạp = 6)
  python pair_riders.py --images data/coco_vn_11cls --out data/coco_vn_paired --src-map identity

  # dataset gốc có data.yaml với tên lớp tiếng Anh (person, motorcycle, bicycle, car...) -> tự ánh xạ theo tên
  python pair_riders.py --images data/bdd_subset --out data/bdd_paired

  # tự chỉ định: gốc 0=person 1=bicycle 2=car 3=motorcycle -> chuẩn
  python pair_riders.py --images X --out Y --src-map "0:8,1:6,2:0,3:4"

LƯU Ý: bộ Roboflow 5 lớp của nhóm KHÔNG có nhãn người -> phải dùng autolabel.py --keep-labels (cần model tìm người).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel import build_parser, run  # noqa: E402

if __name__ == '__main__':
    args = build_parser(no_detect_default=True).parse_args()
    args.no_detect = True
    args.keep_labels = True
    run(args)
