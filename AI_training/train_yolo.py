from ultralytics import YOLO

# Load model YOLO cho object detection (bounding box).
# Lưu ý: Mô hình này được huấn luyện với dữ liệu annotation dạng bounding box.
# Nếu dữ liệu ban đầu của bạn là segmentation (polygon hoặc mask),
# bạn cần chuyển đổi chúng thành bounding box (vd: lấy min/max tọa độ)
# để file annotation có định dạng chuẩn YOLO (class x_center y_center width height, normalized).
model = YOLO("/home/khoa_is_sleep/screws/Robot_control/test_screws.pt")  # Sử dụng model pretrained YOLOv8 nano cho detection

# Huấn luyện mô hình với các tham số đã chọn
model.train(
    data="data.yaml",      # File cấu hình dataset. Đảm bảo trong file này các annotation đã ở định dạng bounding box.
    task="detect",         # Chỉ định nhiệm vụ là detection (bounding box)
    epochs=20,             # Số epoch huấn luyện
    batch=4,               # Batch size
    imgsz=640,             # Kích thước ảnh tiêu chuẩn cho YOLO
    device="cpu",          # Chạy trên CPU (nếu không có GPU, trường hợp này sẽ chạy chậm hơn)
    workers=4,             # Số workers cho dữ liệu (tăng tốc độ xử lý)
    mosaic=1,              # Bật mosaic augmentation
    mixup=0.1,             # Áp dụng nhẹ mixup augmentation
    cache=True,            # Sử dụng cache để train nhanh hơn
    optimizer="SGD",       # Sử dụng SGD optimizer
    lr0=0.01,              # Learning rate ban đầu
    lrf=0.01,              # Learning rate cuối (learning rate final factor)
    cos_lr=True,           # Sử dụng cosine learning rate scheduling
    label_smoothing=0.1,   # Áp dụng label smoothing (giúp giảm overfitting)
    patience=5,            # Early stopping patience
    save=True,             # Lưu model weights sau training
    save_period=5,         # Lưu model sau mỗi 5 epoch
    nbs=64,                # Nominal batch size (nếu cần điều chỉnh learning rate theo batch)
    overlap_mask=False,    # Tắt overlap mask vì ta chỉ quan tâm đến bounding box
    multi_scale=True,      # Huấn luyện với multi-scale inputs giúp tăng hiệu quả
    rect=True              # Sử dụng rectangular training (có thể cải thiện hiệu quả với một số dataset)
)

print("Huấn luyện xong!")
