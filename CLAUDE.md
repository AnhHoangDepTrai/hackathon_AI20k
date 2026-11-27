# PROJECT SYSTEM RULES: VIETNAM ROBOTAXI INSTANCE SEGMENTATION (YOLO26N-SEG)

## 1. BỐI CẢNH & RÀNG BUỘC CỨNG CỦA CUỘC THI
- **Mục tiêu:** Bài toán Data-Centric AI cho nhận diện giao thông Việt Nam (Instance Segmentation).
- **Mô hình cố định:** `yolo26n-seg`, kích thước ảnh đầu vào cố định `imgsz=640`.
- **Nguyên tắc tối thượng:** KHÔNG thay đổi kiến trúc mạng mô hình. Mọi cải thiện về độ chính xác (mAP) phải đến từ **chất lượng dữ liệu, độ chuẩn xác của polygon mask, cân bằng lớp và chiến lược augmentation**.
- **Định dạng nhãn đầu ra bắt buộc:** Ultralytics YOLO-seg (`.txt`). Mỗi dòng là một instance:
  `<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...`
  (Tọa độ đa giác chuẩn hóa về `[0, 1]`, tối thiểu 3 điểm polygon, tuyệt đối không xuất bounding box 4 tọa độ. Ảnh không có đối tượng nào thì để tệp `.txt` rỗng).
- **Cấu trúc nộp dữ liệu chuẩn (`du_lieu.zip`):**
  ```text
  du_lieu.zip
  ├── images/     # Chứa ảnh (.jpg/.png)
  ├── labels/     # Một tệp .txt cho mỗi ảnh
  └── data.yaml   # Tùy chọn cấu hình lớp
  ```
- **Tài liệu hướng dẫn chính thức:** [Antco AI Labeling Guideline](https://competition.antco.ai/data/labeling_guideline.md)

---

## 2. BẢNG 11 LỚP ĐỐI TƯỢNG CHUẨN & QUY TẮC PHÂN LỚP CHÍNH THỨC
Mọi script tự động hóa (Auto-label, Remap, Augmentation) và quy trình kiểm duyệt (QC) bắt buộc tuân thủ đúng ID và định nghĩa sau:

- `0: Car` (Ô tô con): Sedan, SUV, minivan, xe 7 chỗ, bán tải, taxi, **xe điện du lịch**, **xe cứu thương dạng van**, **xe 9–15 chỗ**.
- `1: Bus` (Xe buýt / xe khách): Xe buýt, xe khách **từ 16 chỗ trở lên**. (Lưu ý: Xe 9–15 chỗ thuộc lớp `0: Car`).
- `2: Truck` (Xe tải): Xe tải các cỡ, xe ben, xe bồn, **xe cứu hỏa**, **đầu kéo (CHỈ phần cabin và thân đầu kéo)**.
- `3: Other vehicle` (Phương tiện khác): Ba gác máy, xích lô, xe lam, xe công trình, xe máy cày, **rơ-moóc và sơ-mi rơ-moóc (kể cả thùng container)**, mọi phương tiện không thuộc lớp khác.
- `4: Motorcycle` (Xe máy không người): Xe máy, xe tay ga, xe điện 2 bánh kiểu xe máy, không có người ngồi (**đang đỗ hoặc đang dắt**).
- `5: Motorcyclist` (Xe máy có người): **[QUY TẮC GỘP BẮT BUỘC]** Xe máy có ít nhất một người ngồi (kể cả đang dừng đèn đỏ, người ngồi sau, trẻ em): **1 instance duy nhất phủ cả xe và mọi người trên xe** (bao gồm cả hàng hóa buộc kèm / áo mưa). Tuyệt đối không tách người ngồi sau thành `Pedestrian`.
- `6: Bicycle` (Xe đạp không người): Xe đạp, xe đạp điện kiểu xe đạp, không có người (**đang đỗ hoặc đang dắt**). Xe đạp điện dạng xe máy điện (không có bàn đạp) thuộc `Motorcycle` hoặc `Motorcyclist`.
- `7: Cyclist` (Xe đạp có người): **[QUY TẮC GỘP BẮT BUỘC]** Xe đạp có người đi: **1 instance duy nhất phủ cả xe đạp và người**.
- `8: Pedestrian` (Người đi bộ): Mọi người không ngồi trên phương tiện: đi bộ, đứng, ngồi, trẻ em, **người dắt xe** (người dắt là `Pedestrian`, xe được dắt là `Motorcycle` hoặc `Bicycle` — hai instance riêng biệt).
- `9: Sign` (Biển báo): Biển báo hiệu đường bộ, biển chỉ dẫn, biển công trường. **Chỉ phủ mặt biển, không phủ cột**. Biển quảng cáo, biển cửa hàng **không gán**. Cụm nhiều biển gắn cùng cột: vẽ từng mask riêng cho mỗi mặt biển.
- `10: Traffic light` (Đèn giao thông): **Cả hộp đèn tín hiệu** (xe cộ, người đi bộ), không chỉ bóng đang sáng. Mỗi hộp là một instance. Đèn xe và đèn trang trí **không gán**. KHÔNG phủ cột đèn hay cần vươn.

---

## 3. QUY ƯỚC GÁN NHÃN & NGUYÊN TẮC VẼ MASK

1. **Mask chỉ phủ phần nhìn thấy, không chồng lấn (Zero-Overlap):**
   - Vật ở phía trước giữ pixel, vật phía sau chỉ phủ phần diện tích còn nhìn thấy được.
   - Tuyệt đối không phủ bóng đổ, hình ảnh phản chiếu trên mặt đường/kính, hoặc quầng ánh sáng đèn pha.
   - Vật bị che cắt ngang bởi cột điện, thân cây hoặc dây điện vẫn được tính là một instance duy nhất.
2. **Đầu kéo và Rơ-moóc (Container):**
   - Phần đầu kéo (cabin và khung đầu kéo): gán `2: Truck`.
   - Phần rơ-moóc hoặc sơ-mi rơ-moóc (kể cả thùng container phía sau): gán `3: Other vehicle`. Đây là **hai instance riêng biệt**, không gộp chung!
3. **Người dắt xe máy / xe đạp:**
   - Người dắt xe: gán `8: Pedestrian`.
   - Xe đang được dắt: gán `4: Motorcycle` hoặc `6: Bicycle`.
4. **Đối tượng KHÔNG GÁN (Ignore):**
   - Vật nằm phía sau cửa kính cửa hàng.
   - Người ngồi bên trong ô tô nhìn xuyên qua kính xe.
   - Hình ảnh người/xe vẽ trên áp phích, pano quảng cáo, thân xe buýt.
   - Động vật (chó, mèo, gia súc...).
   - Xe đẩy tay (xe đẩy hàng rong, xe đẩy em bé).
   - Đối tượng quá nhỏ: **cạnh ngắn nhỏ hơn ~1.25% cạnh dài của ảnh** (ví dụ với ảnh 640x640: đối tượng có kích thước nhỏ hơn 8 pixel).
5. **Cây quyết định khi phân vân (Decision Tree):**
   - **Xe 2 bánh:** Có người ngồi trên xe $\rightarrow$ `Motorcyclist (5)` / `Cyclist (7)`. Không người (đang đỗ hoặc đang dắt) $\rightarrow$ `Motorcycle (4)` / `Bicycle (6)`.
   - **Xe 4 bánh trở lên:**
     - Dưới khoảng 9 chỗ hoặc từ 9–15 chỗ $\rightarrow$ `0: Car`.
     - Kiểu xe buýt / xe khách từ 16 chỗ trở lên $\rightarrow$ `1: Bus`.
     - Xe tải, xe ben, xe bồn, xe cứu hỏa, cabin đầu kéo $\rightarrow$ `2: Truck`.
     - Còn lại (rơ-moóc, container, ba gác, xích lô, xe công trình...) $\rightarrow$ `3: Other vehicle`.

---

## 4. YÊU CẦU BẮT BUỘC ĐỐI VỚI AI HỖ TRỢ NGƯỜI VIẾT TOOL (TOOL & PIPELINE ENGINEER)
Khi người dùng yêu cầu viết code (auto-labeling, SAM 2, remap dataset, tiling, copy-paste augmentation, convert định dạng CVAT/YOLO), AI phải tuân thủ nghiêm ngặt:

1. **Bảo toàn Polygon & Tự động ghép cặp Người + Xe:**
   - Mọi pipeline Auto-label (YOLO + SAM 2) bắt buộc phải có bước kiểm tra giao thoa (IoU / containment) giữa `person` và `motorcycle`/`bicycle`:
     - Nếu phát hiện người ngồi trên xe $\rightarrow$ tự động gộp mask thành lớp `5: Motorcyclist` hoặc `7: Cyclist`.
     - Nếu phát hiện người đứng bên cạnh dắt xe $\rightarrow$ tách thành `8: Pedestrian` và `4: Motorcycle` / `6: Bicycle`.
   - Không bao giờ để sót mask của `person` trùng đè bên trong mask của `Motorcyclist`/`Cyclist`.
2. **Xử lý rơ-moóc và đầu kéo:**
   - Trong các pipeline remap/auto-label, nếu phát hiện xe container / xe tải đầu kéo kéo rơ-moóc, phải tách thành 2 polygon riêng (`2: Truck` cho cabin đầu kéo và `3: Other vehicle` cho phần rơ-moóc/container).
3. **Lọc nhiễu & Ngưỡng kích thước cho `imgsz=640`:**
   - Tự động loại bỏ các polygon bị suy biến (`len(points) < 3`), tọa độ nằm ngoài `[0, 1]`, hoặc diện tích mask/kích thước quá nhỏ (`cạnh ngắn < 1.25%` tương đương `< 8px` trên ảnh 640x640).
   - Khi thực hiện Tiling (cắt ảnh 2x2 có chồng lấn) hoặc Copy-Paste augmentation, bắt buộc phải cắt gọt (clip) lại tọa độ polygon theo biên ảnh mới và loại bỏ các phần mask bị che khuất (Zero-Overlap).
4. **Chống rò rỉ dữ liệu (Zero Data Leakage):**
   - Mọi script chia tập dữ liệu hoặc gộp dataset mới bắt buộc phải chạy kiểm tra trùng lặp ảnh (perceptual hash `pHash` / `dHash`) so với tập `golden_val`. Cấm tuyệt đối việc đưa ảnh trùng hoặc gần trùng từ `golden_val` vào `train`.
5. **Tự động ghi nhận Metadata cho báo cáo:**
   - Mọi script tải, lọc hoặc chuyển đổi dữ liệu công khai (Public Dataset) phải tự động xuất kèm file log JSON/CSV ghi rõ: tên dataset, URL, license, số lượng ảnh giữ lại, và bảng ánh xạ class về 11 lớp chuẩn.

---

## 5. YÊU CẦU BẮT BUỘC ĐỐI VỚI AI HỖ TRỢ NGƯỜI QC (QUALITY ASSURANCE & AUDIT)
Khi người dùng yêu cầu viết script kiểm tra nhãn, rà soát lỗi trên CVAT, hoặc đánh giá chất lượng dataset trước khi đóng gói phiên bản mới (`v1`, `v2`, `v3`), AI phải tập trung bắt các lỗi sau:

1. **Kiểm tra lỗi logic phân lớp đặc thù:**
   - Báo lỗi ngay nếu phát hiện 1 instance `8: Pedestrian` có tỷ lệ giao thoa diện tích (IoA) > `0.3` với `5: Motorcyclist`, `7: Cyclist` (lỗi quên gộp người ngồi trên xe).
   - Cảnh báo nếu phát hiện `Truck` bao trùm cả container phía sau (lỗi chưa tách rơ-moóc thành `Other vehicle`).
   - Cảnh báo nếu xe cứu thương van hoặc xe điện du lịch bị gán nhầm vào `Other vehicle` thay vì `Car`.
2. **Kiểm tra hình học Polygon & Chất lượng Mask:**
   - Quét toàn bộ file `.txt` để phát hiện: polygon chỉ có đúng 4 đỉnh tạo thành hình chữ nhật hoàn hảo (nghi ngờ gán nhầm BBox thay vì Polygon), polygon tự cắt nhau (self-intersecting), hoặc nhiều mask chồng lấn lên nhau (vi phạm quy tắc Zero-Overlap).
3. **Kiểm tra cột biển báo / đèn:**
   - Phát hiện các mask thuộc lớp `9: Sign` hoặc `10: Traffic light` có tỷ lệ khung hình (aspect ratio `height/width`) quá mảnh bất thường (`> 4.0`) — dấu hiệu vẽ nhầm cả cột sắt chống biển báo/cột đèn vào mask.
4. **Báo cáo phân bố & Cảnh báo mất cân bằng lớp (Class Imbalance):**
   - Trước khi xuất bản dataset mới, xuất bảng thống kê số lượng instance của cả 11 lớp và cảnh báo đỏ nếu các lớp hiếm (`3: Other vehicle`, `6: Bicycle`, `7: Cyclist`, `10: Traffic light`) có tỷ lệ tăng trưởng thấp hoặc thiếu mẫu trong điều kiện khó (đêm, mưa, kẹt xe).

---

## 6. QUY TẮC PHẢN HỒI CỦA AI TRONG DỰ ÁN NÀY
- Luôn kiểm tra tính nhất quán với 11 ID lớp ở Mục 2 trước khi sinh code hoặc nhận xét dữ liệu.
- Nếu phát hiện yêu cầu của người dùng vi phạm quy tắc gộp lớp (`5`, `7`), quy tắc tách rơ-moóc (`3`), hoặc làm rò rỉ `golden_val`, AI phải **cảnh báo ngay lập tức** và đề xuất phương án sửa đúng chuẩn thay vì âm thầm làm theo.
