from ultralytics import YOLO
import torch

# Đường dẫn file cấu hình dữ liệu
data_yaml = "/home/khoa_is_sleep/screws/AI_training/data.yaml"

# Kiểm tra CPU
device = "cpu"
print(f"Using device: {device}")

# Tải mô hình YOLOv8 segmentation
model = YOLO("yolov8n-seg.pt")  # Sử dụng mô hình segmentation thay vì detection

# Huấn luyện mô hình
model.train(
    data=data_yaml,       # File cấu hình dataset
    epochs=30,           # Tăng số epoch để mô hình học tốt hơn
    batch=2,              # Batch nhỏ vì train trên CPU
    imgsz=840,            # Kích thước ảnh
    device=device,        # Train trên CPU
    task="segment",       # Chỉ định nhiệm vụ là segmentation
    conf=0.001,           # Ngưỡng confidence thấp để không bỏ sót vật thể
    iou=0.45,             # Điều chỉnh IoU threshold
    cache=True,           # Cache dữ liệu để tăng tốc
    workers=0,            # Tránh lỗi đa luồng
    hsv_h=0.015,          # Tăng cường dữ liệu (màu sắc)
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,           # Lật ngẫu nhiên theo chiều dọc
    fliplr=0.5,           # Lật ngẫu nhiên theo chiều ngang
    degrees=10,           # Xoay ảnh ngẫu nhiên
    translate=0.1,        # Dịch ảnh ngẫu nhiên
    scale=0.5             # Phóng to/thu nhỏ ảnh ngẫu nhiên
)

# Xuất mô hình đã huấn luyện
model.export(format="onnx")
