import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional
import matplotlib.pyplot as plt
import random  # Thêm import để sử dụng random

class RotatedDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 morph_kernel_size: int = 3, tracking: bool = True,
                 resize_factor: float = 0.3, skip_frames: int = 2):
        """
        Khởi tạo detector với model YOLO.

        Args:
            model_path (str): Đường dẫn đến model YOLO đã huấn luyện.
            conf_threshold (float): Ngưỡng confidence cho các detection.
            morph_kernel_size (int): Kích thước kernel dùng trong xử lý hình ảnh.
            tracking (bool): Bật/tắt chế độ tracking.
            resize_factor (float): Hệ số giảm kích thước của mỗi frame.
            skip_frames (int): Số frame bỏ qua giữa các lần xử lý.
        """
        self.model = self._load_model(model_path)
        self.conf_threshold = conf_threshold
        self.morph_kernel_size = morph_kernel_size
        self.tracking = tracking
        self.resize_factor = resize_factor
        self.skip_frames = skip_frames
        self.class_names = list(self.model.names.values())
        self.colors = {i: tuple(np.random.randint(0, 255, 3).tolist())
                       for i in range(len(self.class_names))}

    def _load_model(self, model_path: str) -> YOLO:
        """Load model YOLO với xử lý lỗi và ép chạy trên CPU."""
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["PYTHONWARNINGS"] = "ignore"
            return YOLO(model_path, task='detect')
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _get_rotated_bbox(self, img: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Tính toán rotated bounding box sử dụng Otsu thresholding."""
        x1, y1, x2, y2 = bbox.astype(int)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        except cv2.error:
            return None

        kernel = np.ones((self.morph_kernel_size,)*2, np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.bitwise_not(cleaned)
        contours = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        box = np.intp(cv2.boxPoints(rect))
        box[:, 0] += x1
        box[:, 1] += y1
        return box

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Xử lý một frame: detection + tracking + rotated bbox.

        Returns:
            processed_img (np.ndarray): Ảnh sau khi annotate rotated boxes.
            detection_data (list): Danh sách dict chứa info mỗi detection.
        """
        frame_resized = cv2.resize(
            frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor
        )
        results = self.model.track(
            frame_resized, persist=self.tracking,
            conf=self.conf_threshold, verbose=False, iou=0.5
        )
        processed_img = frame_resized.copy()
        detection_data = []

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return processed_img, detection_data

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        for box, score, cls_id, track_id in zip(
            boxes, scores, cls_ids, track_ids
        ):
            if score < self.conf_threshold:
                continue

            label = self.class_names[cls_id]
            color = self.colors.get(cls_id, (255, 255, 255))
            rotated_box = None
            if score > 0.4:
                rotated_box = self._get_rotated_bbox(frame_resized, box)
                if rotated_box is not None:
                    cv2.drawContours(
                        processed_img, [rotated_box], 0, (0, 255, 255), 2
                    )

            detection_data.append({
                "label": label,
                "score": float(score),
                "track_id": int(track_id) if track_id is not None else None,
                "bbox": box.tolist(),
                "rotated_bbox": rotated_box.tolist() if rotated_box is not None else None
            })

        return processed_img, detection_data

    def process_video(self, video_path: str) -> None:
        """
        Xử lý video và hiển thị kết quả liên tục bằng matplotlib.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video {video_path}")

        print(f"Processing video: {video_path}")
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.get_current_fig_manager().set_window_title('Rotated Detector - Press Q to quit')
        im = None
        frame_count = 0
        process_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_count += 1
                if (frame_count - 1) % self.skip_frames != 0:
                    continue

                process_count += 1
                processed_frame, detection_data = self.process_frame(frame)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                print(f"\nFrame {frame_count} (processed {process_count}):")
                if detection_data:
                    for d in detection_data:
                        center_coords = None
                        if d["rotated_bbox"]:
                            pts = np.array(d["rotated_bbox"])
                            cx = int(np.mean(pts[:, 0]))
                            cy = int(np.mean(pts[:, 1]))
                            center_coords = (cx, cy)
                        print(f"  ID: {d['track_id']}, Center: {center_coords}")
                else:
                    print("  Không có detection nào")

                if im is None:
                    im = ax.imshow(frame_rgb)
                    plt.axis('off')
                else:
                    im.set_data(frame_rgb)

                plt.title(f"Frame: {frame_count} (Process: {process_count}) - Press Q to quit")
                plt.draw()
                plt.pause(0.001)

                if plt.waitforbuttonpress(0.001):
                    break
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        finally:
            cap.release()
            plt.close(fig)
            print(f"\nTổng cộng đã xử lý {process_count}/{frame_count} frames")

    def _calculate_custom_center(self, pts, label):
        """
        Tính toán tâm tùy chỉnh dựa trên label.
        Đối với screw2, chia theo chiều dài thành 3 phần và lấy tâm ở 1/3 ngoài cùng.
        """
        # Tính toán tâm thông thường
        default_cx = int(np.mean(pts[:, 0]))
        default_cy = int(np.mean(pts[:, 1]))
        
        # Nếu không phải screw2, trả về tâm thông thường
        if label != "screw2":
            return default_cx, default_cy
            
        # Với screw2, tính toán tâm theo yêu cầu
        # Tìm chiều dài và chiều rộng của rotated box
        rect = cv2.minAreaRect(pts)
        (center_x, center_y), (width, height), angle = rect
        
        # Đảm bảo width là chiều dài (lớn hơn)
        length = max(width, height)
        width = min(width, height)
        
        # Tính toán vector chỉ hướng của chiều dài
        if width != height:  # Chỉ khi không phải hình vuông
            if width < height:  # Nếu chiều cao > chiều rộng
                angle += 90  # Điều chỉnh góc
                
        # Chuyển góc từ độ sang radian
        angle_rad = np.deg2rad(angle)
        
        # Vector đơn vị dọc theo chiều dài
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # Quyết định ngẫu nhiên lấy tâm ở phần 1/3 bên trái hoặc bên phải
        direction = 1 if random.random() > 0.5 else -1
        
        # Tính toán tâm mới: từ tâm hiện tại dịch chuyển 1/3 chiều dài
        # theo hướng đã chọn
        new_cx = int(center_x + direction * (length/3) * dx)
        new_cy = int(center_y + direction * (length/3) * dy)
        
        return new_cx, new_cy

    def process_images_3style(self, folder_path: str) -> None:
        """
        Xử lý tất cả ảnh trong thư mục và hiển thị ba ảnh riêng biệt:
        - Ảnh có bounding box
        - Ảnh có rotated box
        - Ảnh nhị phân
        """
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        files = [os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(valid_ext)]
        if not files:
            print(f"No images in {folder_path}")
            return

        for file in sorted(files):
            img = cv2.imread(file)
            if img is None:
                print(f"Cannot read {file}")
                continue

            img_small = cv2.resize(img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
            rotated_img, dets = self.process_frame(img)

            # Bounding Box Image
            img_bbox = img_small.copy()
            for det in dets:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cls_index = self.class_names.index(det["label"])
                color = self.colors.get(cls_index, (0, 255, 0))
                cv2.rectangle(img_bbox, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_bbox, det["label"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Rotated Box Image
            img_rotated = img_small.copy()
            used_centers = []
            for det in dets:
                if det["rotated_bbox"] is not None:
                    pts = np.array(det["rotated_bbox"], np.int32).reshape(-1, 2)
                    
                    # Sử dụng phương thức tính tâm tùy chỉnh
                    cx, cy = self._calculate_custom_center(pts, det["label"])

                    # Tránh vẽ trùng label quá gần nhau
                    if any(np.linalg.norm(np.array([cx, cy]) - np.array(p)) < 10 for p in used_centers):
                        continue
                    used_centers.append((cx, cy))

                    cls_index = self.class_names.index(det["label"])
                    color = self.colors.get(cls_index, (0, 255, 255))

                    cv2.drawContours(img_rotated, [pts], 0, color, 2)
                    cv2.circle(img_rotated, (cx, cy), 4, (0, 0, 255), -1)
                    label_text = f"{det['label']} ({cx},{cy})"
                    cv2.putText(img_rotated, label_text, (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Binary Mask
            gray_full = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            binary_mask = np.zeros_like(gray_full)
            for det in dets:
                x1, y1, x2, y2 = map(int, det["bbox"])
                roi_gray = gray_full[y1:y2, x1:x2]
                _, bin_roi = cv2.threshold(cv2.GaussianBlur(roi_gray, (3, 3), 0),
                                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_mask[y1:y2, x1:x2] = bin_roi

            # Hiển thị
            cv2.imshow("Bounding Boxes", img_bbox)
            cv2.imshow("Rotated Boxes", img_rotated)
            cv2.imshow("Binary Mask", binary_mask)
            print(f"Processing {file} - press any key to next, q to quit")
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()
    def process_rotated_only(self, folder_path: str) -> None:
        """
        Process and display images showing only Rotated Boxes + Center + Label + Angle.
        Ignore objects that are mostly yellow.
        """
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        files = [os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(valid_ext)]
        if not files:
            print(f"No images in {folder_path}")
            return

        for file in sorted(files):
            img = cv2.imread(file)
            if img is None:
                print(f"Cannot read {file}")
                continue

            img_small = cv2.resize(img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
            _, dets = self.process_frame(img)

            if not dets:
                print(f"No detections for {file}")
                continue

            img_rotated = img_small.copy()
            used_centers = []

            for det in dets:
                if det["rotated_bbox"] is not None:
                    pts = np.array(det["rotated_bbox"], np.int32).reshape(-1, 2)
                    cx, cy = self._calculate_custom_center(pts, det["label"])

                    # Crop ROI from resized image to check for yellow color
                    x1, y1 = np.min(pts, axis=0)
                    x2, y2 = np.max(pts, axis=0)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, img_small.shape[1]), min(y2, img_small.shape[0])
                    roi = img_small[y1:y2, x1:x2]

                    # Check yellow color in ROI
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    yellow_lower = np.array([20, 100, 100])
                    yellow_upper = np.array([30, 255, 255])
                    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
                    yellow_ratio = cv2.countNonZero(yellow_mask) / (roi.shape[0] * roi.shape[1] + 1e-5)

                    if yellow_ratio > 0.1:
                        continue  # Skip yellow objects

                    if any(np.linalg.norm(np.array([cx, cy]) - np.array(p)) < 10 for p in used_centers):
                        continue
                    used_centers.append((cx, cy))

                    # Calculate angle
                    rect = cv2.minAreaRect(pts)
                    angle = rect[2]
                    if angle < -45:
                        angle = 90 + angle

                    cls_index = self.class_names.index(det["label"])
                    color = self.colors.get(cls_index, (0, 255, 255))

                    cv2.drawContours(img_rotated, [pts], 0, color, 2)
                    cv2.circle(img_rotated, (cx, cy), 4, (0, 0, 255), -1)

                    # Add angle information to label
                    label_text = f"{det['label']} ({cx},{cy}) {angle:.1f}°"
                    cv2.putText(img_rotated, label_text, (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Show image
            cv2.imshow("Rotated Boxes Only", img_rotated)
            print(f"Processing {file} - press any key to continue, q to quit")
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()


    def detect_single_image(self, image_path: str, show_display: bool = True) -> tuple:
        """
        Phát hiện đối tượng trong một ảnh được chỉ định bởi đường dẫn.
        
        Args:
            image_path (str): Đường dẫn đến file ảnh cần xử lý.
            show_display (bool): Hiển thị kết quả nếu True.
            
        Returns:
            tuple: (ảnh đã xử lý, dữ liệu phát hiện)
        """
        # Kiểm tra file tồn tại
        if not os.path.exists(image_path):
            print(f"Không tìm thấy file: {image_path}")
            return None, []
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể đọc ảnh: {image_path}")
            return None, []
            
        # Resize ảnh
        img_small = cv2.resize(img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        
        # Xử lý frame để phát hiện đối tượng
        processed_img, detection_data = self.process_frame(img)
        
        # In thông tin diện tích và góc của các đối tượng
        print(f"\n===== THÔNG TIN CÁC ĐỐI TƯỢNG =====")
        for i, det in enumerate(detection_data):
            if det["rotated_bbox"] is not None:
                # Tính diện tích
                rotated_box = np.array(det["rotated_bbox"])
                area = cv2.contourArea(rotated_box)
                det["area"] = float(area)

                # Tính góc
                rect = cv2.minAreaRect(rotated_box)
                angle = rect[2]
                if angle < -45:
                    angle = 90 + angle
                det["angle"] = float(angle)

                # In thông tin
                print(f"Đối tượng {i+1} - {det['label']} - Diện tích: {area:.2f} pixel² - Góc nghiêng: {angle:.2f}°")
        
        if show_display:
            # Chuẩn bị ảnh hiển thị với rotated boxes
            img_display = img_small.copy()
            used_centers = []
            
            for det in detection_data:
                if det["rotated_bbox"] is not None:
                    pts = np.array(det["rotated_bbox"], np.int32).reshape(-1, 2)
                    cx, cy = self._calculate_custom_center(pts, det["label"])

                    if any(np.linalg.norm(np.array([cx, cy]) - np.array(p)) < 10 for p in used_centers):
                        continue
                    used_centers.append((cx, cy))

                    cls_index = self.class_names.index(det["label"])
                    color = self.colors.get(cls_index, (0, 255, 255))

                    cv2.drawContours(img_display, [pts], 0, color, 2)
                    cv2.circle(img_display, (cx, cy), 4, (0, 0, 255), -1)

                    area = det.get("area", 0)
                    angle = det.get("angle", 0)
                    label_text = f"{det['label']} ({cx},{cy}) - {area:.0f}px² - {angle:.1f}°"
                    cv2.putText(img_display, label_text, (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            # Hiển thị kết quả
            cv2.imshow("Detected Objects", img_display)
            print(f"Phát hiện {len(detection_data)} đối tượng trong ảnh: {image_path}")
            print("Nhấn phím bất kỳ để tiếp tục...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return processed_img, detection_data



if __name__ == "__main__":
    model_path = "/home/khoa_is_sleep/screws/last (1).pt"
    data_dir   = "/home/khoa_is_sleep/screws/DATA/Data_screws(origin)/images"

    detector = RotatedDetector(
        model_path,
        resize_factor=0.3,
        skip_frames=1,
        tracking=False
    )
    
    # Sử dụng hàm mới để phát hiện đối tượng trong một ảnh
    image_path = "/home/khoa_is_sleep/screws/testing.png"
    detector.detect_single_image(image_path) 
