import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load mô hình YOLO đã train
model_path = "/home/khoa_is_sleep/screws/runs/segment/train2/weights/last.pt"
model = YOLO(model_path)

# Đọc video đầu vào
video_path = "/home/khoa_is_sleep/screws/Data_screws/test.mp4"
capture = cv2.VideoCapture(video_path)

plt.ion()  # Bật chế độ interactive mode

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    
    # Dự đoán bằng YOLO
    results = model.track(frame, persist=True, tracker="botsort.yaml")

    # Nếu có segmentation
    if results[0].masks is not None:
        for mask, box in zip(results[0].masks.data, results[0].boxes):
            label = int(box.cls[0].item())

            # Chọn màu theo label
            color = np.array([255, 0, 0]) if label == 0 else np.array([0, 255, 0])  # Đỏ cho label 0, xanh lá cho label 1
            alpha = 0.3  # Độ trong suốt của mask

            # Chuyển mask sang NumPy và resize cho khớp frame
            mask = mask.cpu().numpy().squeeze().astype(np.uint8) * 255
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Áp dụng màu lên vùng mask
            mask_bool = mask.astype(bool)
            frame[mask_bool] = frame[mask_bool] * (1 - alpha) + color * alpha

    # Hiển thị bằng matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.pause(0.01)  # Dừng một chút để hiển thị
    plt.clf()  # Xóa frame cũ

capture.release()
plt.close()
