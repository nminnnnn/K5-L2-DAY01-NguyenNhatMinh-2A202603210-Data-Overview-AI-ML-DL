# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**06/11/2026

**Runtime Colab:** GPU (CUDA, T4)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? Record này nói về toàn bộ bức ảnh, lý do là vì trả về kết quả là bảng xếp hạng các lớp cho các ảnh đầu vào
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách lớp thuộc taxonomy đi kèm checkpoint, cụ thể là ImageNet-1K.
Theo mình nghĩ thì model sẽ không tự tạo ra tên lớp mới cho mỗi lần chạy
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Tại 3 loại thông tin này khác nhau, giúp dễ phân biệt hơn.
Ví dụ đối với class_id: đây là mã định danh
class_name: tên lớp,....
Nếu chỉ lưu là cab thì người kiểm tra sẽ bị thiếu thông tin
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
guideline cần phải mô tả thật rõ ràng cách chọn nhãn khi ảnh chứa nhiều đối tượng. Ví dụ có thể yêu cầu ưu tiên đối với những đối tượng chính dựa trên độ nổi bật, hình dạng,...
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score được tạo bởi mô hình, còn Ground Truth là nhãn tham chiếu được xác lập theo quy trình gán nhãn và kiểm tra. Và đặc biệt là prediction có score cao vẫn có thể sai. Score không thay thế việc kiểm tra ảnh và áp dụng guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record. 
Mình chọn đối tượng person, là người đầu bếp đứng quay lưng khu vực bên phải ảnh kitchen 
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.


1. Chép một record: `class_name`=person ; `score`= 0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding Box của người đầu bếp có góc trên bên trái tại tọa độ (385.33, 69.24) và góc dưới bên phải tại (498.92, 348.92). 
Bounding Box có chiều rộng 113.58 pixel và chiều cao 279.68 pixel, được xác định trên ảnh gốc có kích thước 640 × 427 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ confidence threshold, Reviewer cần kiểm tra nhiều prediction hơn, nhất là box có score thấp, để phát hiện false positive, sai class, box trùng hoặc lệch vị trí, kích thước. Reviewer vẫn phải xem toàn bộ ảnh để tìm vật thể bị bỏ sót (false negative) và đối chiếu với ảnh thực tế trước khi kết luận nhãn đúng hay sai.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Mỗi vật thể thuộc taxonomy và đủ điều kiện gán nhãn cần một box `xyxy` riêng, ôm sát phần nhìn thấy và hạn chế lấn sang nền hoặc vật thể khác. Không gộp các vật thể cùng class. Với vật thể bị che hoặc cắt bởi khung ảnh, làm theo guideline và chỉ ước lượng phần khuất khi guideline yêu cầu.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trong ảnh `kitchen`, model dự đoán `person` (score `0.610931`, box `xyxy` `[0.12, 263.18, 61.55, 310.78]`). Người bị cắt ở mép trái ảnh (**truncation**). Mình sẽ làm theo guideline. Nếu điều kiện gán nhãn hoặc cách vẽ box chưa rõ, mình sẽ hỏi Lab Coach để thống nhất cách xử lý.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Bounding Box bao đối tượng bằng hình chữ nhật, còn polygon bám sát đường viền, giúp tách vật thể khỏi nền chính xác hơn.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
Instance ID phân biệt từng vật thể, kể cả khi cùng class. Class ID xác định loại vật thể, còn Tracking ID giữ định danh qua các khung hình. Trong lab ảnh tĩnh, kitchen-001 là Instance ID của một vật thể trong ảnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ mask riêng cho từng vật thể, bám sát phần nhìn thấy, tránh lấn sang nền hoặc vật thể khác. Khi vật thể bị che hoặc đường biên không rõ, làm theo guideline. Nếu chưa có quy định, hỏi Lab Coach trước khi vẽ.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trong ảnh kitchen, model dự đoán dining table với score 0.578416. Mình sẽ kiểm tra mép mask quanh bát, đĩa và dụng cụ để mask chỉ bao gồm phần mặt bàn nhìn thấy. Nếu guideline chưa rõ về vùng bị che, mình sẽ hỏi Lab Coach để thống nhất cách gán nhãn.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một class cho toàn ảnh.|Ảnh traffic có nhiều xe nhưng Top-1 là cab (0.510915).|Chọn nhãn theo guideline.|Kiểm tra class và taxonomy.|
| Phát hiện vật thể |Một class và box cho mỗi vật thể.|person (0.610931) bị cắt ở mép trái ảnh kitchen.|Vẽ box, xử lý truncation.|Kiểm tra box, nhãn và vật thể bị sót.|
| Instance segmentation |Một class và mask cho mỗi instance.|Mask dining table (~0.58) bị các vật trên bàn che một phần.|Vẽ mask sát biên, tách instance.|Kiểm tra biên mask, vùng thừa/thiếu.|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng dữ liệu trong phạm vi bài lab, không tiết lộ thông tin cá nhân, dữ liệu nhạy cảm hoặc chia sẻ dữ liệu khi chưa được phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Mình sẽ dừng xử lý ngay và báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
