import cv2
import os
import albumentations as A
import numpy as np
import shutil
import hashlib

# Chỉ giữ lại các augmentation làm mờ và nhiễu
transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 9), p=0.5),               # Làm mờ ảnh bằng Gaussian Blur
    A.MotionBlur(blur_limit=(7, 15), p=0.4),                # Tạo hiệu ứng motion blur
    A.GaussNoise(p=0.6),                                  # Thêm nhiễu Gaussian
    A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.5), p=0.4),  # Thêm nhiễu ISO
    A.ImageCompression(p=0.3),                             # Nén ảnh tạo hiệu ứng nén, có thể làm giảm độ sắc nét
    A.PixelDropout(dropout_prob=0.02, p=0.3),              # Dropout ngẫu nhiên một số pixel
    A.Downscale(p=0.2),                                  # Giảm độ phân giải, tạo hiệu ứng mờ
    A.Blur(blur_limit=(3, 7), p=0.3),                      # Làm mờ ảnh với blur thông thường
    A.MultiplicativeNoise(multiplier=(0.7, 1.3), per_channel=True, p=0.4),  # Thêm nhiễu nhân, ảnh hưởng đến từng kênh màu riêng
], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

# Định nghĩa thư mục gốc
image_dir = "/home/khoa_is_sleep/screws/DATA/Data_screws(origin)/images"
label_dir = "/home/khoa_is_sleep/screws/DATA/Data_screws(origin)/labels"  # Chứa segmentation labels (YOLO format)
aug_image_dir = "/home/khoa_is_sleep/screws/DATA/Data_screws_aug/aug_images"
aug_label_dir = "/home/khoa_is_sleep/screws/DATA/Data_screws_aug/aug_labels"

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Xử lý từng ảnh trong thư mục gốc
for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.rsplit(".", 1)[0] + ".txt")

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi: Không thể đọc ảnh {image_path}")
            continue
        height, width, _ = image.shape

        # Đọc label (Segmentation) với định dạng YOLO
        polygons = []
        category_ids = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                points = list(map(float, parts[1:]))

                # Chuyển từ YOLO format (chuẩn hóa) sang tọa độ pixel
                polygon = []
                for i in range(0, len(points), 2):
                    x = points[i] * width
                    y = points[i+1] * height
                    polygon.append((x, y))

                polygons.append(polygon)
                category_ids.append(class_id)

        # Tạo 5 ảnh tăng cường cho mỗi ảnh gốc
        for i in range(5):
            # Gộp các keypoints (điểm trên polygon) vào một list duy nhất
            keypoints = [pt for poly in polygons for pt in poly]
            augmented = transform(image=image, keypoints=keypoints)
            aug_image = augmented["image"]
            aug_keypoints = augmented["keypoints"]

            # Chuyển lại keypoints thành danh sách polygon
            aug_polygons = []
            idx = 0
            for poly in polygons:
                aug_polygons.append(aug_keypoints[idx:idx+len(poly)])
                idx += len(poly)

            # Lưu ảnh tăng cường
            aug_filename = f"{filename.rsplit('.', 1)[0]}_aug{i}.jpg"
            aug_image_path = os.path.join(aug_image_dir, aug_filename)
            cv2.imwrite(aug_image_path, aug_image)
            print(f"Đã lưu ảnh: {aug_image_path}")

            # Lưu label tương ứng (chuyển về YOLO format: chuẩn hóa pixel theo width, height)
            aug_label_path = os.path.join(aug_label_dir, aug_filename.replace(".jpg", ".txt"))
            with open(aug_label_path, "w") as f:
                for poly, class_id in zip(aug_polygons, category_ids):
                    bbox_str = " ".join([f"{x/width:.6f} {y/height:.6f}" for x, y in poly])
                    f.write(f"{class_id} {bbox_str}\n")

# Hợp nhất dữ liệu gốc và dữ liệu tăng cường vào thư mục data_train
image_dest = "/home/khoa_is_sleep/screws/DATA/data_train/images"
label_dest = "/home/khoa_is_sleep/screws/DATA/data_train/labels"

os.makedirs(image_dest, exist_ok=True)
os.makedirs(label_dest, exist_ok=True)

def get_file_hash(file_path):
    """Tạo hash MD5 của file để kiểm tra trùng lặp nội dung."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def merge_folders(src_folder1, src_folder2, dest_folder):
    """Hợp nhất hai thư mục mà không bị trùng file."""
    existing_files = {}
    for file in os.listdir(dest_folder):
        file_path = os.path.join(dest_folder, file)
        if os.path.isfile(file_path):
            existing_files[get_file_hash(file_path)] = file
    
    for src_folder in [src_folder1, src_folder2]:
        for file in os.listdir(src_folder):
            src_path = os.path.join(src_folder, file)
            if os.path.isfile(src_path):
                file_hash = get_file_hash(src_path)
                if file_hash not in existing_files:
                    shutil.copy2(src_path, dest_folder)
                    existing_files[file_hash] = file
                    print(f"Copied: {file}")
                else:
                    print(f"Skipped (duplicate): {file}")

# Hợp nhất ảnh
print("Merging images...")
merge_folders(image_dir, aug_image_dir, image_dest)

# Hợp nhất nhãn
print("Merging labels...")
merge_folders(label_dir, aug_label_dir, label_dest)

print("Merging completed!")
