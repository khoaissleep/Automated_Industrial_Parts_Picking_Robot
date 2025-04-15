import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# Đường dẫn đến thư mục ảnh và nhãn
image_dir = 'Data_screws/images'
label_dir = 'Data_screws/labels'

# Hàm tạo màu sắc khác nhau cho mỗi class ID
def random_color(class_id):
    random.seed(class_id)  # Đảm bảo mỗi class có màu cố định
    return [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]  # Màu sáng

def draw_segmentation(image_path, label_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image.copy()  # Tạo lớp overlay để vẽ màu trong suốt
    h, w, _ = image.shape

    # Đọc file nhãn
    if not os.path.exists(label_path):
        print(f"Lỗi: Không tìm thấy label {label_path}")
        return
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])  # Nhãn lớp
        polygon = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)  # Lấy danh sách tọa độ

        # Chuyển tọa độ từ tỷ lệ YOLO sang pixel
        polygon[:, 0] *= w  # Scale x theo chiều rộng ảnh
        polygon[:, 1] *= h  # Scale y theo chiều cao ảnh
        polygon = polygon.astype(int)

        # Chọn màu theo class_id
        color = random_color(class_id)

        # Vẽ đường viền segmentation
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)

        # Tô màu trong suốt lên overlay
        cv2.fillPoly(overlay, [polygon], color=color)

    # Pha trộn overlay với ảnh gốc để tạo hiệu ứng trong suốt
    alpha = 0.4  # Độ trong suốt (0.0 - hoàn toàn trong, 1.0 - hoàn toàn che phủ)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Hiển thị ảnh
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# --- Xử lý xóa file không có file đối ứng ---

# Kiểm tra các file trong thư mục ảnh: nếu không tìm thấy file label tương ứng thì xóa ảnh
for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Chỉ lấy file ảnh
        image_path = os.path.join(image_dir, filename)
        # Giả sử file label có đuôi .txt và cùng tên với file ảnh
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        if not os.path.exists(label_path):
            print(f"Không tìm thấy label cho ảnh {image_path}. Đang xóa ảnh...")
            os.remove(image_path)
        else:
            print(f"Kiểm tra ảnh: {image_path}")
            draw_segmentation(image_path, label_path)

# Kiểm tra các file trong thư mục label: nếu không tìm thấy file ảnh tương ứng thì xóa label
for filename in os.listdir(label_dir):
    if filename.lower().endswith(".txt"):
        label_path = os.path.join(label_dir, filename)
        # Kiểm tra các định dạng ảnh: jpg, png, jpeg
        base_name = os.path.splitext(filename)[0]
        image_paths = [
            os.path.join(image_dir, base_name + ext) 
            for ext in [".jpg", ".png", ".jpeg"]
        ]
        # Nếu không tồn tại file ảnh nào thì xóa file label
        if not any(os.path.exists(ip) for ip in image_paths):
            print(f"Không tìm thấy ảnh cho label {label_path}. Đang xóa label...")
            os.remove(label_path)
