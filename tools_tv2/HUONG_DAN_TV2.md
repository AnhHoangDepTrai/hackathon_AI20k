# HƯỚNG DẪN BỘ TOOL THÀNH VIÊN 2 — Auto-Label & Dữ liệu tổng hợp

> Đọc hết mục 0 và mục 1 (5 phút) rồi làm theo **mục 2 (luồng chạy)**. Mục 3–6 là chi tiết từng tool, đọc khi cần.

---

## 0. Bộ tool này làm gì? (giải thích như cho người mới)

Mô hình bị khoá cứng là `yolo26n-seg`, nên **điểm chỉ tăng khi DỮ LIỆU tốt hơn**. Bộ tool này tự động hoá 4 việc tốn công nhất:

| # | Tool | Làm gì | Cần GPU? |
|---|------|--------|----------|
| 1 | `autolabel.py` | Máy **tự vẽ nhãn nháp**: YOLO11x tìm vật → YOLO-World tìm biển báo / xích lô / container → **SAM 2** vẽ polygon sát viền từng vật | **Có** (Colab/Kaggle) |
| 2 | `pair_riders.py` (+ có sẵn bên trong autolabel) | **Ghép người với xe**: người ngồi trên xe máy → gộp thành 1 polygon lớp **5**; người dắt xe → người lớp **8**, xe lớp **4** (xe đạp tương tự 7 / 6) | Không |
| 3 | `copy_paste.py` | **Nhân bản lớp hiếm** (3 Other vehicle, 6 Bicycle, 7 Cyclist): cắt vật từ ảnh này dán sang ảnh khác (ưu tiên nền đêm/mưa), tự sửa nhãn | Không |
| 4 | `tile_images.py` | **Cắt ảnh lớn thành 4 ô** chồng lấn 20% để biển báo / đèn nhỏ không bị mất khi resize về 640 | Không |
| + | `visualize.py` | Vẽ nhãn lên ảnh để **nhìn bằng mắt** trước khi train | Không |
| + | `check_leak.py` | Kiểm tra ảnh trong train có **trùng golden_val** không | Không |

Mọi tool đều tự động tuân theo `CLAUDE.md`:
- đúng 11 id lớp, xuất **polygon YOLO-seg** (không xuất bbox),
- **Zero-Overlap**: vật phía trước giữ pixel, vật phía sau bị trừ phần bị che (kể cả lỗ bên trong — dùng đường cắt "lỗ khoá"),
- bỏ vật **quá nhỏ** (cạnh ngắn < 1.25% cạnh dài ảnh, hoặc < 64 px khi ảnh về 640 — đúng ngưỡng hệ thống chấm bỏ qua),
- bỏ **người ngồi trong ô tô**, bỏ **nắp capo xe dashcam** bị gán nhầm là Car, xoá **"gai"** (đường chéo dài do polygon lỗi),
- tự ghi `run_log.json` / `*_log.json` (tham số, ánh xạ lớp, số lượng) để **TV1 đưa vào slide báo cáo**.

### Phát hiện quan trọng về bộ Roboflow hiện có (756 ảnh)
Mình đã mở và vẽ thử nhãn bộ `Vietnam Traffic Vehicle Detectio.v1i.yolo26`:
1. **Thứ tự lớp gốc**: `0 bicycle, 1 bus, 2 car, 3 motorcycle, 4 truck` → tool tự đọc `data.yaml` và ánh xạ đúng (car→0, bus→1, truck→2, motorcycle→4/5, bicycle→6/7). **Đừng** dùng id gốc trực tiếp!
2. Polygon xe máy **chỉ phủ thân xe, không phủ người lái** và **không có nhãn người nào** → bắt buộc chạy `autolabel.py --keep-labels` (có model tìm người) mới tách được 4/5. Chạy `pair_riders.py` trên bộ này là **vô ích** (tool sẽ in cảnh báo đỏ).
3. Có **336 dòng là bbox 5 cột** (không phải polygon) → tool đổi thành polygon hình chữ nhật và ghi vào `review.csv` để người sửa. Khuyên bật `--refine-human-masks` để SAM vẽ lại cho sát.
4. Có **39 polygon lỗi** (đường kẻ chéo dài xuyên ảnh) và **6 ảnh gán nắp capo xe mình là Car** → tool tự sửa/bỏ.
5. Không có Pedestrian / Sign / Traffic light / Other vehicle → autolabel bổ sung.
6. Ảnh đã là 640x640 có viền đen → **không cần tile**; tool tự né viền đen khi copy-paste.

---

## 1. Cài đặt (1 lần)

**Máy cá nhân (Windows)** — cho pair_riders / copy_paste / tile / visualize / check_leak:
```bash
cd hackathon_AI20k/tools_tv2
pip install -r requirements.txt
python tests/test_tv2.py            # phải in "TẤT CẢ KIỂM THỬ ĐẠT"
python tests/test_tile_cp.py        # phải in "TILING + COPY-PASTE ĐẠT"
```

**Colab / Kaggle (GPU)** — cho autolabel: mở `colab_autolabel.ipynb`, chạy từ trên xuống (notebook tự cài `ultralytics==8.4.214`, tự tải yolo11x / yolov8x-worldv2 / sam2.1_b). Tool cần nằm trên GitHub của nhóm (push thư mục `tools_tv2`) **hoặc** nén `tools_tv2.zip` để lên Drive.

> Đường dẫn có dấu cách thì **luôn bọc trong dấu ngoặc kép** `"..."`.

---

## 2. LUỒNG CHẠY CHUẨN (làm theo đúng thứ tự)

```
 [Ảnh mới của TV3 / bộ Roboflow]
          │
          ▼
 ① autolabel.py  (GPU)  ──► nhãn nháp 11 lớp + review.csv + viz/
          │
          ▼
 ② visualize.py + TV4 sửa trên CVAT theo review_priority.csv (ảnh nghi ngờ nhất trước)
          │
          ▼
 ③ check_leak.py  (loại ảnh trùng golden_val)  ──► TRAIN SẠCH
          │
          ├──► ④ tile_images.py  (chỉ ảnh lớn ≥1280px có biển/đèn nhỏ)
          └──► ⑤ copy_paste.py   (nhân bản lớp 3/6/7 lên nền đêm/mưa)
          │
          ▼
 ⑥ Gộp: train sạch + tiles + copy-paste  ──► đưa TV5 train   (golden_val để NGUYÊN, không tile, không paste)
```

### Bước ① — Auto-label

**a) Bộ Roboflow có sẵn (giữ nhãn người vẽ, bổ sung phần thiếu):** — trên Colab:
```bash
python tools_tv2/autolabel.py --images "/content/drive/MyDrive/AI20k/Vietnam Traffic Vehicle Detectio.v1i.yolo26" \
       --out /content/drive/MyDrive/AI20k/roboflow_v1_11cls --keep-labels --refine-human-masks --resume
```
**b) Ảnh mới chưa có nhãn (frame dashcam của TV3):**
```bash
python tools_tv2/autolabel.py --images /content/drive/MyDrive/AI20k/frames_moi --out /content/drive/MyDrive/AI20k/draft_v2 --resume
```
**c) Active learning — khi TV5 đã có `best.pt` 11 lớp**: dùng chính model của đội làm detector (nó đã biết Motorcyclist/Cyclist/Sign…):
```bash
python tools_tv2/autolabel.py --images frames_moi --out draft_v3 --det-model best.pt --open-vocab world --resume
```
Thử trước 20 ảnh bằng `--limit 20`, xem thư mục `viz/`, ổn rồi mới chạy hết.

Kết quả trong `--out`:
| Tệp | Ý nghĩa |
|---|---|
| `.../images`, `.../labels` | ảnh + nhãn 11 lớp (giữ cấu trúc train/valid/test nếu có) |
| `data.yaml` | 11 lớp chuẩn — import CVAT dạng "Ultralytics YOLO Segmentation" |
| `review.csv` | từng ca máy **không chắc** (lý do, lớp, toạ độ) |
| `review_priority.csv` | danh sách ảnh xếp theo độ nghi ngờ → TV4 sửa từ trên xuống |
| `run_log.json` | ánh xạ lớp, số lượng trước/sau, model đã dùng → **dán vào slide** |
| `viz/` | ảnh trái (gốc) / phải (nhãn) để xem nhanh |

Tốc độ tham khảo trên T4: ~1–2 giây/ảnh (yolo11x@1280 + World + SAM2.1-b). 756 ảnh ≈ 15–25 phút. Colab ngắt → chạy lại đúng lệnh có `--resume`, nó làm tiếp.

### Bước ② — Kiểm tra bằng mắt
```bash
python visualize.py --data roboflow_v1_11cls --review roboflow_v1_11cls/review_priority.csv --n 60
```
Mở thư mục `viz_check/`. Bảng cuối in số instance mỗi lớp (lớp < 100 bị đánh dấu `<-- ÍT`).

### Bước ③ — Chống rò rỉ golden_val (BẮT BUỘC trước khi tile/paste)
```bash
python check_leak.py --train roboflow_v1_11cls --golden golden_val/images --quarantine leak_quarantine
```
Ảnh trùng/gần trùng bị **di chuyển** (không xoá) sang `leak_quarantine/`.

### Bước ④ — Tiling (chỉ khi có ảnh độ phân giải cao)
```bash
python tile_images.py --data data/frames_hd_labeled/train --out data/tiles_v2
```

### Bước ⑤ — Copy-Paste lớp hiếm
```bash
python copy_paste.py --src data/coco_bike_11cls data/roboflow_v1_11cls/train --bg data/roboflow_v1_11cls/train data/night_rain_frames \
       --out data/cp_v2 --classes 3,6,7 --n 400 --bg-mode night --night-fx 0.3 --rain-fx 0.3 --golden-val golden_val/images
```
**Mở `data/cp_v2/bank_preview.jpg` trước tiên**: đây là toàn bộ vật sẽ được dán. Vật nào mask xấu (lẹm, dính nền, thực ra là người đang đạp mà gán Bicycle…) → nguồn đó kém, đổi nguồn rồi chạy lại.

### Bước ⑥ — Gộp cho TV5
Notebook train nhận 1 thư mục có `images/train`, `images/val`, `labels/train`, `labels/val`. Chép tiles + copy-paste vào **train**, golden_val vào **val**. Không để notebook tự chia ngẫu nhiên (tile/paste của 1 ảnh có thể rơi vào cả train và val = rò rỉ).

---

## 3. Chi tiết `autolabel.py` — máy hoạt động thế nào

Với mỗi ảnh:
1. **Đọc nhãn có sẵn** (`--keep-labels`), ánh xạ lớp theo tên trong `data.yaml` (in bảng ánh xạ ra màn hình — **kiểm tra bảng này!**).
2. **Làm sạch nhãn người vẽ**: xoá gai polygon, bỏ nắp capo, đổi bbox 5 cột → polygon (ghi review). Nếu `--refine-human-masks`: SAM vẽ lại mask, chỉ thay khi khớp IoU ≥ 0.5.
3. **YOLO11x** (chạy ở 1280 để bắt vật nhỏ) tìm: người, xe đạp, ô tô, xe máy, buýt, tải, đèn, biển STOP.
4. **YOLO-World** tìm theo câu chữ: "traffic sign", "road sign", "cyclo", "tricycle", "three-wheeled cargo cart", "excavator", "shipping container", "semi-trailer"…
5. **SAM 2** nhận từng hộp → vẽ mask sát viền (mask bị cắt trong hộp nới 10% để SAM không "tràn" ra mặt đường).
6. **Bỏ trùng** với nhãn có sẵn (xe 2 bánh dùng ngưỡng lỏng vì nhãn người vẽ chỉ phủ thân xe).
7. "Other vehicle" của YOLO-World nếu đè lên xe COCO → **không thêm**, ghi review (tránh biến xe máy thường thành ba gác).
8. **Rơ-moóc**: container nằm trong hộp Truck → mặc định chỉ ghi review `truck_maybe_has_trailer_check_split`; thêm `--split-trailer` thì tự tách (Truck = cabin, Other vehicle = container).
9. **Bỏ người trong ô tô** (mask người nằm trong mask xe, chân cao hơn đáy xe).
10. **Ghép cặp người–xe** (mục 4) + SAM trên hộp hợp người+xe để lấy cả hàng hoá buộc sau yên / tà áo mưa.
11. **Zero-Overlap** theo chiều sâu (đáy vật càng thấp = càng gần camera = ở trước), lọc vật nhỏ, xuất polygon.

Tham số hay dùng:
| Tham số | Khi nào dùng |
|---|---|
| `--keep-labels` | dataset đã có nhãn (Roboflow, public) |
| `--src-map "0:6,1:1,2:0,3:4,4:2"` | data.yaml thiếu/sai tên lớp → tự chỉ định gốc:chuẩn (-1 = bỏ) |
| `--add-classes 8,9,10` | chỉ cho máy THÊM người/biển/đèn (xe giữ nguyên của người vẽ) |
| `--open-vocab gdino` | biển báo bị sót nhiều → GroundingDINO chính xác hơn, chậm hơn ~2 lần |
| `--sam-model sam2.1_l.pt` | cần mask đẹp nhất (chậm hơn) |
| `--mask-source det --det-model yolo11x-seg.pt` | gấp thời gian: bỏ SAM cho lớp COCO |
| `--split-trailer` | tự tách container khỏi đầu kéo (nhớ duyệt lại) |
| `--limit 20` | chạy thử |
| `--resume` | chạy tiếp sau khi Colab ngắt |

Các lý do trong `review.csv`:
| reason | Người duyệt cần làm |
|---|---|
| `pair_borderline`, `pair_ambiguous_2_vehicles` | xem người có thật sự ngồi trên xe đó không |
| `person_near_two_wheeler_kept_as_pedestrian` | người đứng sát xe nhưng máy cho là đi bộ/dắt xe — kiểm tra |
| `low_conf_auto_label` | vật máy không chắc (conf < 0.5): đúng thì giữ, sai thì xoá |
| `maybe_other_vehicle_check_class` | có thể là ba gác/xích lô bị nhận là xe máy/xe tải |
| `truck_maybe_has_trailer_check_split`, `truck_or_trailer_ambiguous`, `trailer_split_from_truck` | kiểm tra tách đầu kéo / rơ-moóc |
| `bbox_line_converted_to_rectangle_polygon` | nhãn gốc là hộp → vẽ lại polygon |
| `person_on_other_vehicle_merged` | người ngồi trên xích lô/ba gác đã gộp vào Other vehicle |
| `ego_hood_dropped`, `human_polygon_spike_removed`, `person_in_vehicle_dropped` | đã tự sửa, chỉ cần liếc qua |
| `sam_failed` | SAM không ra mask → vẽ tay |

---

## 4. Chi tiết ghép cặp người–xe (`pair_riders.py` / bên trong autolabel)

Với mỗi cặp (người, xe 2 bánh) máy đo hình học hộp + mask:
- **chồng ngang**: người ngồi thì hộp người phủ gần trọn bề ngang xe,
- **lệch tâm**: người ngồi nằm giữa xe; người dắt đứng ở đầu/bên hông xe,
- **chân**: đáy hộp người phải nằm sâu trong hộp xe (30%–135% chiều cao xe),
- **đầu**: đỉnh người cao hơn (hoặc ngang) đỉnh xe,
- **tỉ lệ**: người cao 0.45–4 lần xe,
- **chạm nhau**: mask người (nới vài px) chạm mask xe.

Đạt hết → **ngồi trên xe** → hợp mask người + xe thành 1 polygon lớp 5/7 (mỗi xe tối đa 4 người với xe máy, 2 với xe đạp — chở con vẫn đúng). Người đứng thẳng, chân chạm đất cạnh xe, lệch tâm → **dắt xe** → giữ 8 + 4/6.
Người ngồi trên ba gác/xích lô (lớp 3) → gộp vào lớp 3 (vì "người ngồi trên phương tiện không phải Pedestrian"), có ghi review.

Dùng riêng `pair_riders.py` cho dataset **đã có cả nhãn người và xe** (COCO, BDD100K… sau khi TV3 remap):
```bash
python pair_riders.py --images data/coco_vn --out data/coco_vn_paired            # tự ánh xạ theo tên trong data.yaml
python pair_riders.py --images data/x --out data/y --src-map identity            # đã là 11 lớp chuẩn
```
BDD100K có lớp `rider` → tool coi là người (8) rồi ghép vào xe → 5/7. Chạy trên CPU, ~20 ảnh/giây.

---

## 5. Chi tiết `copy_paste.py`

1. **Ngân hàng vật** (từ `--src`): chỉ lấy vật nguyên vẹn (không chạm mép ảnh/viền đen), cao ≥ 24px, mask không quá rỗng, không vỡ > 3 mảnh. Xem `bank_preview.jpg`.
2. **Nền** (từ `--bg`): `--bg-mode night` tự đo độ sáng, chỉ lấy ảnh tối (< `--night-thr` 0.28). Ảnh mưa: đưa thư mục ảnh mưa vào `--bg`. Không đủ nền đêm → `--bg-mode any --night-fx 0.5` (biến nền ngày thành đêm giả).
3. **Vị trí + kích thước theo phối cảnh**: học "pixel/mét theo độ cao y" từ chính xe/người có sẵn trong ảnh nền → xe đạp ở xa thì nhỏ, ở gần thì to. Không phóng to quá 1.6 lần. Không dán đè quá 50% một vật cũ.
4. **Hoà màu** LAB theo vùng nền (`--harmonize 0.5`), làm mờ nếu vật sắc hơn nền, thêm nhiễu giống nền, viền mềm phía trong mask.
5. **Nhãn**: vật dán ở trên cùng; vật cũ bị che được trừ phần bị che (Zero-Overlap); vật cũ không bị che giữ nguyên polygon gốc.
6. `--rain-fx 0.3`: 30% ảnh có thêm vệt mưa (chỉ trong vùng ảnh, không lên viền đen).
7. Ghi `manifest.jsonl` (mỗi ảnh dán gì, từ đâu, tỉ lệ bao nhiêu) + `copy_paste_log.json` cho báo cáo.

Tham số: `--n` số ảnh tạo · `--per-image 1 3` số vật mỗi ảnh · `--max-reuse 12` mỗi vật nguồn dán tối đa 12 lần (chống học thuộc) · `--seed 27` (dùng seed của đội để tái lập).

---

## 6. Chi tiết `tile_images.py`

Ảnh W×H, lưới 2×2, chồng lấn o=0.2 → mỗi ô rộng `W/(2−o)` (1920×1080 → 4 ô 1067×600). Vật nằm ở vùng chồng lấn xuất hiện ở cả 2 ô (có lợi). Polygon được raster rồi cắt đúng theo biên ô; mảnh còn < 30% diện tích **và** < 400 px (thang 640) bị bỏ; vật quá nhỏ bị bỏ.
Mặc định chỉ tile ảnh cạnh dài ≥ 1280 **và** có Sign/Traffic light nhỏ (< 40px sau khi về 640). Muốn tile mọi ảnh lớn: `--require-classes all`. Chồng lấn 15%: `--overlap 0.15`.

---

## 7. LƯU Ý SỐNG CÒN (đọc kỹ)

1. **Nhãn máy là NHÁP.** Luôn xem `viz/` và sửa theo `review_priority.csv` trước khi đưa train. Không có thời gian sửa hết thì ít nhất sửa 20% ảnh đứng đầu danh sách.
2. **golden_val bất khả xâm phạm**: không autolabel-rồi-dùng-luôn cho val mà không duyệt, không tile, không copy-paste, không dùng làm nguồn/nền. Luôn truyền `--golden-val` cho copy_paste/tile và chạy `check_leak.py`.
3. **Ảnh tổng hợp ≤ 20–30% tập train.** Nhiều hơn dễ làm model học "kiểu ảnh dán" thay vì đường phố thật. So điểm val của TV5 trước và sau khi thêm để quyết định giữ hay bỏ.
4. **Nguồn copy-paste phải sạch.** Bộ Roboflow có 7 xe đạp (một số thực ra có người đạp) và polygon thô → đừng dùng làm nguồn chính; lấy từ COCO (đã remap + pair_riders) hoặc từ output autolabel đã duyệt.
5. **Kiểm tra bảng ánh xạ lớp** in ra đầu mỗi lần chạy `--keep-labels`. Sai 1 dòng = hỏng cả dataset.
6. **Đừng chạy `pair_riders.py` trên bộ Roboflow** (không có nhãn người) — dùng autolabel `--keep-labels`.
7. **Đầu kéo + rơ-moóc**: tự tách (`--split-trailer`) có thể nhầm xe tải thùng thường. Ít thời gian → để mặc định (chỉ đánh dấu), TV4 tách tay những ảnh bị đánh dấu.
8. **Không ghi đè dữ liệu gốc**: mọi tool ghi ra `--out` mới; `--out` trùng `--images` bị từ chối.
9. **Giữ lại mọi `run_log.json`, `copy_paste_log.json`, `tile_log.json`, `manifest.jsonl`** — đó là số liệu cho slide (bộ dữ liệu, giấy phép, số ảnh, cách ánh xạ 11 lớp, các vòng cải thiện).
10. Thời gian còn ít: ưu tiên ① autolabel Roboflow + ② duyệt nhanh → giao TV5 train baseline sớm; copy-paste/tile làm sau khi đã có điểm mốc để so.

---

## 8. Sự cố thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không thấy ảnh nào` | sai đường dẫn / thiếu dấu ngoặc kép quanh đường dẫn có dấu cách |
| Bảng ánh xạ ghi `BỎ (không biết lớp này)` | dùng `--src-map "id_gốc:id_chuẩn,..."` |
| `CUDA out of memory` | `--sam-model sam2.1_t.pt` hoặc `--det-imgsz 960` |
| YOLO-World lỗi tải CLIP | `--open-vocab none` (mất biển báo/xích lô) hoặc `--open-vocab gdino` |
| `Ngân hàng instance rỗng` | `--src` chưa có lớp cần dán ở dạng nguyên vẹn → thêm nguồn |
| `Không ảnh nào được tile` | ảnh nhỏ hơn 1280px (bình thường với Roboflow) — hoặc `--min-side 1000`, `--require-classes all` |
| Kiểm thử | `python tests/test_tv2.py`, `tests/test_pipeline_mock.py`, `tests/test_tile_cp.py`, `tests/test_fake_ultralytics.py` |
