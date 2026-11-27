# PROJECT SYSTEM RULES: VIETNAM ROBOTAXI INSTANCE SEGMENTATION (YOLO26N-SEG)

## 1. BỐI CẢNH & RÀNG BUỘC CỨNG CỦA CUỘC THI
- **Mục tiêu:** Bài toán Data-Centric AI cho nhận diện giao thông Việt Nam (Instance Segmentation).
- **Mô hình cố định:** `yolo26n-seg`, kích thước ảnh đầu vào cố định `imgsz=640`.
- **Nguyên tắc tối thượng:** KHÔNG thay đổi kiến trúc mạng mô hình. Mọi cải thiện về độ chính xác (mAP) phải đến từ **chất lượng dữ liệu, độ chuẩn xác của polygon mask, cân bằng lớp và chiến lược augmentation**.
- **Định dạng nhãn đầu ra bắt buộc:** YOLO-seg (`.txt`), mỗi dòng gồm: `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>` (tọa độ chuẩn hóa `[0, 1]`, tối thiểu 3 điểm polygon, tuyệt đối không xuất bounding box 4 tọa độ).

## 2. BẢNG 11 LỚP ĐỐI TƯỢNG & QUY TẮC GÁN NHÃN BẤT DI BẤT DỊCH
Mọi script tự động hóa (Auto-label, Remap, Augmentation) và quy trình kiểm duyệt (QC) bắt buộc tuân thủ đúng ID và định nghĩa sau:

- `0: Car` — Ô tô con (sedan, SUV, MPV, taxi, xe bán tải pickup không chở hàng quá khổ).
- `1: Bus` — Xe buýt nội đô, xe khách từ 16 chỗ trở lên (Ford Transit, Solati, giường nằm, 2 tầng).
- `2: Truck` — Xe tải các loại, xe đầu kéo (gộp cả đầu kéo + rơ-moóc thành 1 mask), xe bán tải chở hàng cồng kềnh.
- `3: Other vehicle` — Xe ba gác máy (xe lôi), xích lô, xe điện du lịch, xe cứu thương, xe cứu hỏa, xe chuyên dụng/tự chế. Gộp chung người lái vào 1 mask duy nhất.
- `4: Motorcycle` — Xe máy, xe tay ga, xe máy điện KHÔNG CÓ NGƯỜI ngồi trên yên.
- `5: Motorcyclist` — **[QUY TẮC GỘP BẮT BUỘC]** 1 polygon duy nhất bao trùm **người lái + người ngồi sau + xe máy + hàng hóa chở kèm + áo mưa**. Tuyệt đối không tách người ngồi sau thành `Pedestrian`.
- `6: Bicycle` — Xe đạp thường, xe đạp điện bàn đạp KHÔNG CÓ NGƯỜI ngồi trên yên.
- `7: Cyclist` — **[QUY TẮC GỘP BẮT BUỘC]** 1 polygon duy nhất bao trùm **người đạp + xe đạp**.
- `8: Pedestrian` — Người đi bộ trên vỉa hè/băng qua đường (bao gồm cả ba lô/túi xách đang đeo). Không gán cho người đã ngồi lên phương tiện.
- `9: Sign` — Chỉ vẽ mặt bảng biển báo giao thông (tròn, tam giác, vuông, biển chỉ dẫn làn/tên đường). KHÔNG vẽ cột chống. Cụm nhiều biển trên 1 cột phải tách thành từng mask riêng.
- `10: Traffic light` — Chỉ vẽ hộp vỏ đèn tín hiệu giao thông (bao gồm cả đèn người đi bộ và bộ đếm giây gắn liền). KHÔNG vẽ cột đèn hay cần vươn.

---

## 3. YÊU CẦU BẮT BUỘC ĐỐI VỚI AI HỖ TRỢ NGƯỜI VIẾT TOOL (TOOL & PIPELINE ENGINEER)
Khi người dùng yêu cầu viết code (auto-labeling, SAM 2, remap dataset, tiling, copy-paste augmentation, convert định dạng CVAT/YOLO), AI phải tuân thủ nghiêm ngặt:

1. **Bảo toàn Polygon & Tự động ghép cặp Người + Xe:**
   - Mọi pipeline Auto-label (YOLO + SAM 2) bắt buộc phải có bước kiểm tra giao thoa (IoU / containment) giữa `person` và `motorcycle`/`bicycle` để tự động gộp mask thành lớp `5` (`Motorcyclist`) hoặc `7` (`Cyclist`) trước khi xuất file `.txt`.
   - Không bao giờ để sót mask của `person` trùng đè bên trong mask của `Motorcyclist`/`Cyclist`.
2. **Lọc nhiễu & Ngưỡng kích thước cho `imgsz=640`:**
   - Tự động loại bỏ các polygon bị suy biến (`len(points) < 3`), tọa độ nằm ngoài `[0, 1]`, hoặc diện tích mask quá nhỏ (`< 10x10 px` khi quy đổi về khung hình `640x640`) để tránh làm nhiễu hàm loss của `yolo26n-seg`.
   - Khi thực hiện Tiling (cắt ảnh 2x2 có chồng lấn) hoặc Copy-Paste augmentation, bắt buộc phải cắt gọt (clip) lại tọa độ polygon theo biên ảnh mới và tính lại diện tích hợp lệ.
3. **Chống rò rỉ dữ liệu (Zero Data Leakage):**
   - Mọi script chia tập dữ liệu hoặc gộp dataset mới bắt buộc phải chạy kiểm tra trùng lặp ảnh (perceptual hash `pHash` / `dHash`) so với tập `golden_val`. Cấm tuyệt đối việc đưa ảnh trùng hoặc gần trùng từ `golden_val` vào `train`.
4. **Tự động ghi nhận Metadata cho báo cáo:**
   - Mọi script tải, lọc hoặc chuyển đổi dữ liệu công khai (Public Dataset) phải tự động xuất kèm file log JSON/CSV ghi rõ: tên dataset, URL, license, số lượng ảnh giữ lại, và bảng ánh xạ class về 11 lớp chuẩn.

---

## 4. YÊU CẦU BẮT BUỘC ĐỐI VỚI AI HỖ TRỢ NGƯỜI QC (QUALITY ASSURANCE & AUDIT)
Khi người dùng yêu cầu viết script kiểm tra nhãn, rà soát lỗi trên CVAT, hoặc đánh giá chất lượng dataset trước khi đóng gói phiên bản mới (`v1`, `v2`, `v3`), AI phải tập trung bắt các lỗi sau:

1. **Kiểm tra lỗi logic phân lớp đặc thù Việt Nam:**
   - Báo lỗi ngay nếu phát hiện 1 instance `8: Pedestrian` có tỷ lệ giao thoa diện tích (IoA) > `0.3` với `4: Motorcycle`, `5: Motorcyclist`, `6: Bicycle`, `7: Cyclist`, hoặc `3: Other vehicle` (lỗi quên gộp người ngồi trên xe).
   - Cảnh báo nếu trong cùng 1 ảnh có nhãn `4: Motorcycle` hoặc `6: Bicycle` nằm giữa lòng đường (nghi ngờ gán nhầm xe đang lưu thông có người lái thành xe không người).
2. **Kiểm tra hình học Polygon & Chất lượng Mask:**
   - Quét toàn bộ file `.txt` để phát hiện: polygon chỉ có đúng 4 đỉnh tạo thành hình chữ nhật hoàn hảo (nghi ngờ gán nhầm BBox thay vì Polygon), polygon tự cắt nhau (self-intersecting), hoặc nhiều mask cùng lớp chồng lấn lên nhau với `IoU > 0.75` (lỗi nhân đôi nhãn - duplicate annotation).
3. **Kiểm tra độ che khuất & Cột biển báo/đèn:**
   - Phát hiện các mask thuộc lớp `9: Sign` hoặc `10: Traffic light` có tỷ lệ khung hình (aspect ratio `height/width`) quá mảnh bất thường (`> 4.0`) — dấu hiệu người gán nhãn vẽ nhầm cả cột sắt chống biển báo/cột đèn vào mask.
4. **Báo cáo phân bố & Cảnh báo mất cân bằng lớp (Class Imbalance):**
   - Trước khi cho phép xuất bản dataset mới, AI QC phải xuất bảng thống kê số lượng instance của cả 11 lớp và cảnh báo đỏ nếu các lớp hiếm (`3: Other vehicle`, `6: Bicycle`, `7: Cyclist`, `10: Traffic light`) có tỷ lệ tăng trưởng thấp hoặc thiếu mẫu trong điều kiện khó (đêm, mưa, kẹt xe).

---

## 5. QUY TẮC PHẢN HỒI CỦA AI TRONG DỰ ÁN NÀY
- Luôn kiểm tra tính nhất quán với 11 ID lớp ở Mục 2 trước khi sinh code hoặc nhận xét dữ liệu.
- Nếu phát hiện yêu cầu của người dùng vi phạm quy tắc gộp lớp (`5`, `7`) hoặc làm rò rỉ `golden_val`, AI phải **cảnh báo ngay lập tức** và đề xuất phương án sửa đúng chuẩn thay vì âm thầm làm theo.