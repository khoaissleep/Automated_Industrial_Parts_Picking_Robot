# Last updated: 16h 19/3
# Robot     COM3
# Conveyor  COM11
# Encoder   COM6

import serial
import serial.tools.list_ports
import sys, os
from pypylon import pylon
import cv2
import time
import numpy as np
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Thêm lớp RotatedDetector từ trackingvideo.py
class RotatedDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 morph_kernel_size: int = 3, tracking: bool = True,
                 resize_factor: float = 1.0, skip_frames: int = 0):
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
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Màu sắc có thể tùy chỉnh

    def _load_model(self, model_path: str) -> YOLO:
        """Load model YOLO."""
        try:
            return YOLO(model_path, task='detect')
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _get_rotated_bbox(self, img: np.ndarray, bbox: np.ndarray):
        """Tính toán rotated bounding box sử dụng Otsu thresholding."""
        x1, y1, x2, y2 = bbox.astype(int)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        try:
            # Sử dụng kích thước nhỏ hơn cho Gaussian Blur để tăng tốc
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            return None

        kernel = np.ones((self.morph_kernel_size,)*2, np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.bitwise_not(cleaned)
        contours = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        return np.intp(cv2.boxPoints(rect)) + [x1, y1]

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Xử lý một frame: thực hiện detection và tracking.
        
        Returns:
            tuple: (processed_image, detection_data)
        """
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        
        results = self.model.track(frame, persist=self.tracking,
                                  conf=self.conf_threshold, verbose=False,
                                  iou=0.5)
        processed_img = frame.copy()
        detection_data = []

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return processed_img, detection_data

        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes.xyxy, 'cpu') else results[0].boxes.xyxy.numpy()
        scores = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes.conf, 'cpu') else results[0].boxes.conf.numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int) if hasattr(results[0].boxes.cls, 'cpu') else results[0].boxes.cls.numpy().astype(int)
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if hasattr(results[0].boxes.id, 'cpu') else results[0].boxes.id.numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        for box, score, cls_id, track_id in zip(boxes, scores, cls_ids, track_ids):
            if score < self.conf_threshold:
                continue

            label = self.class_names[cls_id]
            color = self.colors.get(cls_id, (255, 255, 255))
            
            # Chỉ tính rotated box cho các object có confidence cao
            rotated_box = self._get_rotated_bbox(frame, box)
            if rotated_box is not None:
                cv2.drawContours(processed_img, [rotated_box], 0, (0, 255, 255), 2)
                # Tính tâm của rotated box
                center_x = int(np.mean(rotated_box[:, 0]))
                center_y = int(np.mean(rotated_box[:, 1]))
                # Tính diện tích của rotated box
                area = cv2.contourArea(rotated_box)
                
                # Thêm thông tin vào detection_data
                detection_data.append({
                    "label": label,
                    "score": float(score),
                    "track_id": int(track_id) if track_id is not None else None,
                    "bbox": box.tolist(),
                    "rotated_bbox": rotated_box,
                    "center": (center_x, center_y),
                    "area": area,
                    "class_id": int(cls_id)  # Thêm class_id vào detection_data
                })
                
                # Vẽ tọa độ tâm
                cv2.circle(processed_img, (center_x, center_y), 3, (0, 0, 255), -1)
                text = f"ID:{track_id if track_id is not None else 'N/A'} {label}"
                cv2.putText(processed_img, text, (center_x, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return processed_img, detection_data

velo = 2000                                     # max 3000
acce = 15000                                    # max 50000
conveyor_velo = -20                             # conveyor speed
x_re1, y_re1, z_re1 = 150, -275, -770           # receiver 1 coordinate
x_re2, y_re2, z_re2 = 50, -275, -770            # receiver 2 coordinate
move_up = 40                                    # robot move up after gripping
z_grip = -809

x_scale = 100/127
x_offset = 429
y_scale = 100/153
y_offset = 219.6
conveyor_offset = conveyor_velo / 10            # = velo chia cho hệ số (chưa biết)

serial_port = None
conveyor_serial_port = None
object_tracks = {}
next_id = 1  # Thêm biến global next_id

with open('calibration_parameters.json', 'r') as f:
    calib_data = json.load(f)

camera_matrix = np.array(calib_data['camera_matrix'])
distortion_coefficients = np.array(calib_data['dist_coeffs'])
new_camera_matrix = np.array(calib_data['new_camera_matrix'])

class BaslerCamera:
    def __init__(self):
        self.camera = None
        self.grabbing = False
        self.WIDTH = 1280   # max 1280
        self.HEIGHT = 1024  # max 1024
        self.object_tracker = {}
        self.next_object_id = 0
        self.perspective_transform = None
        self.transform_file = 'camera_transform.json'
        self.load_transform()
        # Thay thế mô hình YOLO bằng RotatedDetector
        self.detector = RotatedDetector('test_screws.pt', conf_threshold=0.4, tracking=True)
        print("Screw detector loaded successfully")

    def start_camera(self):
        """Khởi động camera"""
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.camera.PixelFormat.SetValue("BayerRG8")
            self.camera.Width.SetValue(self.WIDTH)
            self.camera.Height.SetValue(self.HEIGHT)
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.grabbing = True
            return True
        except Exception as e:
            print(f"Could not start camera: {str(e)}")
            return False

    def stop_camera(self):
        """Dừng camera"""
        self.grabbing = False
        # if self.camera.IsGrabbing():
        #     self.camera.StopGrabbing()
        self.camera.Close()

    def get_frame(self):
        """Lấy frame từ camera và áp dụng biến đổi phối cảnh"""
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        if self.camera is not None and self.grabbing:
            # Lấy frame từ camera
            result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                image = converter.Convert(result)
                frame = image.GetArray()  # shape: (H, W, 3) BGR

                # ----------------- Hiệu chỉnh Distortion -----------------------
                frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)

                # ----------------- Perspective Transformation -----------------------
                h, w = 1024, 1280  # Kích thước của ảnh gốc
                corners = np.array([
                    [0, 0, 1],
                    [w, 0, 1],
                    [w, h, 1],
                    [0, h, 1]
                ]).T  # shape: (3, 4)

                M = self.perspective_transform  # Ma trận biến đổi phối cảnh đã có
                transformed_corners = np.dot(M, corners)
                transformed_corners /= transformed_corners[2, :]

                min_x, max_x = transformed_corners[0].min(), transformed_corners[0].max()
                min_y, max_y = transformed_corners[1].min(), transformed_corners[1].max()
                new_width = int(max_x - min_x)
                new_height = int(max_y - min_y)

                T = np.array([[1, 0, -min_x],
                            [0, 1, -min_y],
                            [0, 0, 1]])

                M_adjusted = T @ M
                rectified_frame = cv2.warpPerspective(frame, M_adjusted, (new_width, new_height))

                # ----------------- Cắt và Resize ảnh -----------------------
                y1, y2 = 0, new_height  # giữ full chiều cao
                x1, x2 = 120, 550  # Cắt theo chiều ngang
                rectified_frame = rectified_frame[y1:y2, x1:x2]

                # Resize ảnh để dễ hiển thị
                rectified_frame = cv2.resize(rectified_frame, (640, 640))

                return rectified_frame
            else:
                print("Failed to grab frame")
                return None
        return None

    def load_transform(self):
        """Đọc ma trận transform từ file JSON"""
        try:
            if os.path.exists(self.transform_file):
                with open(self.transform_file, 'r') as f:
                    transform_data = json.load(f)
                self.perspective_transform = np.array(transform_data['matrix'])
                print(f"\nLoaded transform matrix from {self.transform_file}")
                print(f"Calibration timestamp: {transform_data['timestamp']}")
                return True
        except Exception as e:
            print(f"Error loading transform matrix: {e}")
        return False

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Screw Sorter V1.0")
        self.basler_camera = BaslerCamera()
        # Bỏ phần logo để tránh lỗi file không tồn tại
        # logo = tk.PhotoImage(file='deltax_logo.png')
        # self.root.iconphoto(True, logo)
        self.root.geometry("1300x650")

        # Main frame: split into 2 parts
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5, anchor='w')
        self.left_frame.configure(width=500)  # 50% width

        # Right frame (camera)
        self.camera_frame = ttk.LabelFrame(self.main_frame, text="Camera View")
        self.camera_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5, anchor='e')
        self.camera_frame.configure(width=600)  # 50% width

        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Đảm bảo các frame giữ kích thước
        self.main_frame.pack_propagate(False)
        self.left_frame.pack_propagate(False)
        self.camera_frame.pack_propagate(False)

        # Connection frame
        self.connection_frame = ttk.Frame(self.left_frame)
        self.connection_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.com_frame = ttk.LabelFrame(self.connection_frame, text="Robot Connection")
        self.com_frame.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)  # Đặt ở bên trái

        self.com_ports = [port.device for port in serial.tools.list_ports.comports()]
        self.com_var = tk.StringVar()
        self.com_combo = ttk.Combobox(self.com_frame, textvariable=self.com_var, values=self.com_ports, width=26)
        self.com_combo.pack(side=tk.LEFT, padx=5, pady=5)

        self.connect_btn = ttk.Button(self.com_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.conveyor_com_frame = ttk.LabelFrame(self.connection_frame, text="Conveyor Connection")
        self.conveyor_com_frame.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)  # Đặt ở bên phải

        self.conveyor_com_var = tk.StringVar()
        self.conveyor_com_combo = ttk.Combobox(self.conveyor_com_frame, textvariable=self.conveyor_com_var, values=self.com_ports, width=26)
        self.conveyor_com_combo.pack(side=tk.LEFT, padx=5, pady=5)

        self.conveyor_connect_btn = ttk.Button(self.conveyor_com_frame, text="Connect", command=self.toggle_conveyor_connection)
        self.conveyor_connect_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Notebook trong left_frame
        self.notebook = ttk.Notebook(self.left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab Auto Mode
        self.auto_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.auto_tab, text="Auto Mode")

        # Frame controls cho Auto Mode
        self.auto_control_frame = ttk.Frame(self.auto_tab)
        self.auto_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Nút Start/Stop
        self.auto_btn = ttk.Button(self.auto_control_frame, text="START", command=self.toggle_auto_mode)
        self.auto_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.auto_running = False

        # Label hiển thị số hạt đã pick
        self.picked_label = ttk.Label(self.auto_control_frame, text="Picked seeds: 0")
        self.picked_label.pack(side=tk.LEFT, padx=20, pady=5)
        self.picked_count = 0

        # Frame cho Command Window trong Auto Mode
        self.auto_response_frame = ttk.LabelFrame(self.auto_tab, text="Command Window")
        self.auto_response_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.auto_response_text = tk.Text(self.auto_response_frame, height=20)
        self.auto_response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab Manual Mode
        self.manual_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.manual_tab, text="Manual Mode")

        # Frame điều khiển bên trái
        self.control_frame = ttk.LabelFrame(self.manual_tab, text="Controls")
        self.control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Các nút điều khiển
        ttk.Button(self.control_frame, text="Home",
                  command=lambda: self.send_gcode(f"G01 X0 Y0 Z{z_grip + move_up} F300 A1500")).pack(padx=5, pady=5, fill="x")
        ttk.Button(self.control_frame, text="Get Low",
                  command=lambda: self.send_gcode(f"G01 X0 Y0 Z{z_grip} F300 A1500")).pack(padx=5, pady=5, fill="x")
        ttk.Button(self.control_frame, text="Current position",
                  command=lambda: self.send_gcode("Position")).pack(padx=5, pady=5, fill="x")
        ttk.Button(self.control_frame, text="Receiver 1",
                  command=lambda: self.send_gcode(f"G01 X{x_re1} Y{y_re1} Z{z_re1} F300 A1500")).pack(padx=5, pady=5, fill="x")
        ttk.Button(self.control_frame, text="Receiver 2",
                  command=lambda: self.send_gcode(f"G01 X{x_re2} Y{y_re2} Z{z_re2} F300 A1500")).pack(padx=5, pady=5, fill="x")

        # Nút Gripper toggle
        self.gripper_btn = ttk.Button(self.control_frame, text="Gripper: OFF", command=self.toggle_gripper)
        self.gripper_btn.pack(padx=5, pady=5, fill="x")
        self.gripper_state = False  # False = Off, True = On

        # Nút Conveyor toggle
        self.conveyor_btn = ttk.Button(self.control_frame, text="Conveyor: OFF", command=self.toggle_conveyor)
        self.conveyor_btn.pack(padx=5, pady=5, fill="x")
        self.conveyor_state = False  # False = Off, True = On

        # Nút Camera toggle
        self.camera_btn = ttk.Button(self.control_frame, text="Camera: OFF", command=self.toggle_camera)
        self.camera_btn.pack(padx=5, pady=5, fill="x")
        self.camera_state = False  # False = Off, True = On

        # Frame Pick and Place
        self.pp_frame = ttk.LabelFrame(self.control_frame, text="Pick and Place")
        self.pp_frame.pack(padx=5, pady=5, fill="x")
        self.pick_in_progress = False

        # Frame cho tọa độ X
        x_frame = ttk.Frame(self.pp_frame)
        x_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(x_frame, text="X (mm):").pack(side=tk.LEFT)
        self.pp_x = ttk.Entry(x_frame, width=10)
        self.pp_x.pack(side=tk.RIGHT)

        # Frame cho tọa độ Y
        y_frame = ttk.Frame(self.pp_frame)
        y_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(y_frame, text="Y (mm):").pack(side=tk.LEFT)
        self.pp_y = ttk.Entry(y_frame, width=10)
        self.pp_y.pack(side=tk.RIGHT)

        # Frame cho số thứ tự receiver
        box_frame = ttk.Frame(self.pp_frame)
        box_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(box_frame, text="Box (1/2):").pack(side=tk.LEFT)
        self.pp_box = ttk.Entry(box_frame, width=10)
        self.pp_box.pack(side=tk.RIGHT)

        # Nút Run pick and place
        ttk.Button(self.pp_frame, text="Run", command=self.execute_pick_and_place).pack(padx=5, pady=5)

        # Frame GCode
        self.gcode_frame = ttk.LabelFrame(self.control_frame, text="GCode Command")
        self.gcode_frame.pack(padx=5, pady=5, fill="x")

        self.gcode_entry = ttk.Entry(self.gcode_frame)
        self.gcode_entry.pack(padx=5, pady=5, fill="x")
        ttk.Button(self.gcode_frame, text="Send", command=self.send_gcode_from_entry).pack(padx=5, pady=5)

        # Frame cho Command Window trong Manual Mode
        self.response_frame = ttk.LabelFrame(self.manual_tab, text="Command Window")
        self.response_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.response_text = tk.Text(self.response_frame, height=30, width=45)
        self.response_text.pack(padx=5, pady=5)

        # Tab Calibrate
        self.calibrate_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.calibrate_tab, text="Calibrate")

        # Frame controls cho Calibrate Mode
        self.calibrate_control_frame = ttk.Frame(self.calibrate_tab)
        self.calibrate_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Nút Start/Stop và Update
        self.calibrate_btn = ttk.Button(self.calibrate_control_frame, text="START", command=self.toggle_calibrate_mode)
        self.calibrate_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.update_btn = ttk.Button(self.calibrate_control_frame, text="Update", command=self.update_calibration, state='disabled')
        self.update_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Frame cho Command Window trong Calibrate Mode
        self.calibrate_response_frame = ttk.LabelFrame(self.calibrate_tab, text="Command Window")
        self.calibrate_response_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.calibrate_response_text = tk.Text(self.calibrate_response_frame, height=10)
        self.calibrate_response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame cho Matrix Display
        self.matrix_frame = ttk.LabelFrame(self.calibrate_tab, text="Calibration Parameters")
        self.matrix_frame.pack(fill=tk.X, padx=5, pady=5)

        # Thêm label cho Calibration Matrix
        self.calibration_matrix_label = ttk.Label(self.matrix_frame, text="Calibration Matrix: Not calibrated")
        self.calibration_matrix_label.pack(padx=5, pady=5, anchor='w')  # Căn lề trái

        self.last_update_label = ttk.Label(self.matrix_frame, text="Last update: Not calibrated")
        self.last_update_label.pack(padx=5, pady=5, anchor='w')

        # Biến cho calibration
        self.calibrate_running = False
        self.clicked_points = []
        self.real_points = [
            [-80, -80], [0, -80], [80, -80],    # Hàng trên
            [-80, 0],   [0, 0],   [80, 0],      # Hàng giữa
            [-80, 80],  [0, 80],  [80, 80]      # Hàng dưới
        ]

        # Load existing calibration data
        self.load_existing_calibration()

        # Cấu hình grid
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.auto_tab.grid_columnconfigure(1, weight=1)
        self.manual_tab.grid_columnconfigure(1, weight=1)

        # Tạo blank cursor để ẩn con trỏ chuột
        self.blank_cursor = "none"  # Hoặc có thể dùng: "none" trên Windows, "" trên Unix
        self.suppress_response = False  # Thêm biến trạng thái
        self.conveyor_stop_first_time = False # bien dung de skip frame dau

#----------------------------------------------------------------------------------#
#----------------------------------- Connection -----------------------------------#
#----------------------------------------------------------------------------------#
    def toggle_connection(self):
        """Kết nối/ngắt kết nối COM port"""
        global serial_port  # Sử dụng biến serial_port toàn cục

        if serial_port is None or not serial_port.is_open:
            try:
                port = self.com_var.get()
                serial_port = serial.Serial(port, 115200, timeout=1)
                self.connect_btn.config(text="Disconnect")
                self.update_response(f"Connected to DeltaX: {port}")
                self.suppress_response = True
                time.sleep(0.2)
                self.send_gcode("G01 Z-760 F200 A600")
                time.sleep(0.2)
                self.send_gcode("G28")
                self.suppress_response = False
            except Exception as e:
                self.update_response(f"Connection error: {str(e)}")
        else:
            try:
                serial_port.close()
                serial_port = None
                self.connect_btn.config(text="Connect")
                self.update_response("Disconnected")
            except Exception as e:
                self.update_response(f"Disconnection error: {str(e)}")

    def toggle_conveyor_connection(self):
        """Kết nối/ngắt kết nối cổng COM cho băng tải"""
        global conveyor_serial_port  # Sử dụng biến conveyor_serial_port toàn cục

        if conveyor_serial_port is None or not conveyor_serial_port.is_open:
            try:
                port = self.conveyor_com_var.get()
                conveyor_serial_port = serial.Serial(port, 115200, timeout=1)
                self.conveyor_connect_btn.config(text="Disconnect")
                self.update_response(f"Connected to X Conveyor: {port}")
            except Exception as e:
                self.update_response(f"Connection error: {str(e)}")
        else:
            try:
                conveyor_serial_port.close()
                conveyor_serial_port = None
                self.conveyor_connect_btn.config(text="Connect")
                self.update_response("Disconnected from Conveyor")
            except Exception as e:
                self.update_response(f"Disconnection error: {str(e)}")

#----------------------------------------------------------------------------------#
#------------------------------------- G-code -------------------------------------#
#----------------------------------------------------------------------------------#
    def send_gcode(self, command):
        """Gửi GCode tới robot"""
        global serial_port

        if serial_port and serial_port.is_open:
            try:
                serial_port.write((command + "\n").encode())
                if not self.suppress_response:  # Kiểm tra biến trạng thái
                    self.update_manual_response(f"Sent: {command}")
                response = ""
                while True:
                    if serial_port.in_waiting > 0:
                        response += serial_port.read(serial_port.in_waiting).decode()
                        if response:
                            break
                if response and not self.suppress_response:  # Kiểm tra biến trạng thái
                    self.update_manual_response(f"Response: {response}")

            except Exception as e:
                self.update_manual_response(f"Send error: {str(e)}")
        else:
            self.update_manual_response("Not connected to robot")

    def send_gcode_to_conveyor(self, command):
        """Gửi GCode tới conveyor"""
        global conveyor_serial_port  # Sử dụng biến conveyor_serial_port toàn cục

        if conveyor_serial_port and conveyor_serial_port.is_open:
            try:
                conveyor_serial_port.write((command + "\n").encode())
                self.update_manual_response(f"Sent to conveyor: {command}")
            except Exception as e:
                self.update_manual_response(f"Error sending to conveyor: {str(e)}")
        else:
            self.update_manual_response("Not connected to conveyor")

    def send_gcode_from_entry(self):
        """Gửi GCode từ entry box"""
        command = self.gcode_entry.get()
        self.send_gcode(command)

#----------------------------------------------------------------------------------#
#----------------------------------- Manual mode ----------------------------------#
#----------------------------------------------------------------------------------#
    def toggle_gripper(self):
        """Toggle gripper on/off"""
        if not self.gripper_state:  # Nếu đang Off -> chuyển On
            self.send_gcode("M03 D1")
            self.gripper_btn.config(text="Gripper: ON")
            self.gripper_state = True
        else:  # Nếu đang On -> chuyển Off
            self.send_gcode("M05 D1")
            self.gripper_btn.config(text="Gripper: OFF")
            self.gripper_state = False

    def toggle_conveyor(self):
        """Toggle conveyor on/off"""
        if not self.conveyor_state:  # Nếu đang Off -> chuyển On
            self.send_gcode_to_conveyor(f"M311 {conveyor_velo}")  # Bắt đầu băng tải
            self.conveyor_btn.config(text="Conveyor: ON")
            self.conveyor_state = True
        else:  # Nếu đang On -> chuyển Off
            self.send_gcode_to_conveyor("M311 0")  # Dừng băng tải
            self.conveyor_btn.config(text="Conveyor: OFF")
            self.conveyor_state = False

    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_state:
            self.update_manual_response("Opening camera...")
            self.camera_btn.config(text="Camera: ON")
            self.camera_state = True
            self.basler_camera.start_camera()
            self.update_camera_loop()
        else:
            self.update_manual_response("Camera stopped")
            self.camera_btn.config(text="Camera: OFF")
            self.camera_state = False
            self.basler_camera.stop_camera()
            self.camera_label.configure(image='')
            self.camera_label.image = None

    def update_camera_loop(self, enable_grip=False):
        """Cập nhật frame camera liên tục"""
        if self.picked_count > 0: enable_grip = True
        if self.basler_camera.grabbing:  # Gọi get_frame đã cập nhật
            frame = self.basler_camera.get_frame()
            if frame is not None:
                # Detect và xử lý frame với RotatedDetector
                processed_img, detection_data = self.basler_camera.detector.process_frame(frame)
                
                # Tạo overlay để hiển thị
                overlay = processed_img.copy()
                frame_detections = []

                # Xử lý các detection từ detector
                for detection in detection_data:
                    center_x, center_y = detection["center"]
                    area = detection["area"]
                    track_id = detection["track_id"]
                    score = detection["score"]
                    label = detection["label"]
                    # Lấy class_id từ detector nếu có sẵn, nếu không thì thử lấy từ label
                    if "class_id" in detection:
                        cls_id = detection["class_id"]
                    else:
                        try:
                            cls_id = self.basler_camera.detector.class_names.index(label) if label in self.basler_camera.detector.class_names else -1
                        except:
                            cls_id = -1  # Mặc định nếu không xác định được
                    
                    # Chuyển đổi tọa độ sang tọa độ robot
                    cxn = center_y * x_scale - x_offset  # swap - scale - offset x
                    cyn = center_x * y_scale - y_offset  # swap - scale - offset y
                    
                    # Thêm vào danh sách detections để cập nhật track
                    # Bổ sung thêm label vào detection để phân loại
                    frame_detections.append((center_x, center_y, 100, area, cls_id))

                # -------------------------------------
                # (B) Update object tracks với frame_detections
                # -------------------------------------
                self.match_or_create_tracks(frame_detections, max_dist=20)

                self.update_manual_response("", clear=True)
                self.update_manual_response(f"Quantity: {len(frame_detections)}")  # số vật thể trên màn hình
                
                for obj_id, data in object_tracks.items():
                    cx, cy = data['center']
                    cxn = cy * x_scale - x_offset           # swap - scale - offset x
                    cyn = cx * y_scale - y_offset           # swap - scale - offset y
                    area = data.get('area', 0)
                    cls_id = data.get('class_id', -1)
                    label = self.basler_camera.detector.class_names[cls_id] if 0 <= cls_id < len(self.basler_camera.detector.class_names) else "Unknown"
                    
                    text = f"Id:{obj_id} \t Class:{label} \t Area:{area:.0f} \t ({cxn:.1f} ; {cyn:.1f})"
                    self.update_manual_response(text)
                    
                    # Hiển thị ID và tâm của đối tượng
                    color = self.basler_camera.detector.colors.get(cls_id, (0, 0, 255))
                    cv2.putText(overlay, f"Id:{obj_id} {label}", (int(cx)-10, int(cy)-18), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    cv2.circle(overlay, (int(cx), int(cy)), 5, color, -1)

                    #-------------------------------------------------------------------------------------------------#
                    #------------------------------------- Classification output -------------------------------------#
                    #-------------------------------------------------------------------------------------------------#
                    if enable_grip:
                        cx_moving = cxn - conveyor_offset
                        cy_moving = cyn
                        
                        # Phân loại dựa trên label thay vì diện tích
                        if not data.get('picked', False):
                            # Phân loại dựa trên class_id: 0 vào khay 2, 1 vào khay 1
                            # Kiểm tra cls_id hợp lệ trước khi phân loại
                            cls_id = data.get('class_id', -1)
                            if cls_id == 1:
                                box_number = 1  # Ốc vít loại 1 (lớn) vào khay 1
                            elif cls_id == 0:
                                box_number = 2  # Ốc vít loại 0 (nhỏ) vào khay 2
                            else:
                                # Nếu không xác định được loại, mặc định vào khay 1
                                box_number = 1
                                self.update_manual_response(f"Warning: Unknown class ID {cls_id}, default to box 1")
                                
                            self.pick_n_place(cx_moving, cy_moving, box_number)
                            self.picked_count += 1
                            self.picked_label.config(text=f"Picked screws: {self.picked_count}")
                            data['picked'] = True

                # Kết hợp overlay và frame gốc
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, processed_img, 1 - alpha, 0, processed_img)

                # Hiển thị frame
                frame = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo

        # Lập lịch cập nhật frame tiếp theo
        if self.camera_state:
            self.root.after(30, self.update_camera_loop)

    def match_or_create_tracks(self,detections, max_dist=20, alpha=0.1, max_missed_frames= 20):
        """
        detections: list of (cx, cy, closeness, area, class_id)
        For each detection, if an existing track's center is within max_dist,
        update that track. Otherwise, create a new track with a new unique ID.

        alpha: smoothing factor for updating the 'closeness' metric.
        """
        global next_id
        next_id = max(object_tracks.keys()) + 1 if object_tracks else 1
        # Mark all existing tracks as not matched for this frame
        for track in object_tracks.values():
            track['matched'] = False

        # For each detection, try to find an existing track within threshold.
        for (cx, cy, closeness, area, cls_id) in detections:
            updated = False
            for track_id, track in object_tracks.items():
                tx, ty = track['center']
                # Calculate Euclidean distance between detection and track center
                if np.hypot(cx - tx, cy - ty) < max_dist:
                    # Update this track
                    n = track['frames_seen']
                    track['center'] = (cx, cy)
                    track['area'] = area
                    track['class_id'] = cls_id  # Cập nhật class_id
                    # Không cần cập nhật closeness nữa
                    track['frames_seen'] += 1
                    track['missed_frames'] = 0
                    track['matched'] = True
                    updated = True
                    break  # Use the first matching track

            if not updated:
                # Create a new track if no existing track was close enough
                object_tracks[next_id] = {
                    'center': (cx, cy),
                    'area': area,
                    'class_id': cls_id,  # Lưu class_id
                    'frames_seen': 1,
                    'closeness': 100,  # Giá trị mặc định
                    'missed_frames': 0,
                    'matched': True
                }
                next_id += 1

        # Increment missed_frames for tracks that weren't updated this frame,
        # and remove any that exceed max_missed_frames.
        tracks_to_delete = []
        for track_id, track in object_tracks.items():
            if not track.get('matched', False):
                track['missed_frames'] += 1
                if track['missed_frames'] > max_missed_frames:
                    tracks_to_delete.append(track_id)
        for track_id in tracks_to_delete:
            del object_tracks[track_id]

#----------------------------------------------------------------------------------#
#------------------------------------ Auto mode -----------------------------------#
#----------------------------------------------------------------------------------#
    def toggle_auto_mode(self):
        """Toggle auto mode on/off"""
        if not self.auto_running:
            self.update_auto_response("Running auto screw sorter...")
            self.auto_running = True
            self.auto_btn.config(text="STOP")

            self.send_gcode(f"G01 X0 Y0 Z{z_grip + move_up} F{velo} A{acce}")   # Robot home
            self.send_gcode("M05 D1")                                           # Gripper off
            self.send_gcode_to_conveyor(f"M311 {conveyor_velo}")                # Start conveyor
            self.conveyor_state = True
            # Reset picked count
            self.picked_count = 0
            self.picked_label.config(text=f"Picked screws: {self.picked_count}")

            self.camera_state = True
            self.basler_camera.start_camera()
            self.update_camera_loop(enable_grip=True)
        else:
            self.update_auto_response("Stop auto mode")
            self.auto_running = False
            self.auto_btn.config(text="START")

            self.send_gcode(f"G01 X0 Y0 Z{z_grip + move_up} F{velo} A{acce}")   # Robot home
            self.send_gcode("M05 D1")                                           # Gripper off
            self.send_gcode_to_conveyor("M311 0")                               # Stop conveyor
            self.conveyor_state = False
            self.picked_count = 0
            self.picked_label.config(text=f"Picked screws: {self.picked_count}")

            # Tắt camera
            self.camera_state = False
            self.basler_camera.stop_camera()
            self.camera_label.configure(image='')
            self.camera_label.image = None

#----------------------------------------------------------------------------------#
#------------------------------------- Calibrate ----------------------------------#
#----------------------------------------------------------------------------------#
    def load_existing_calibration(self):
        """Load và hiển thị ma trận hiệu chuẩn hiện có"""
        try:
            if os.path.exists(self.basler_camera.transform_file):
                with open(self.basler_camera.transform_file, 'r') as f:
                    data = json.load(f)
                    matrix = np.array(data['matrix'])
                    timestamp = data['timestamp']

                    # Làm tròn ma trận đến 2 chữ số thập phân
                    rounded_matrix = np.round(matrix, 2)

                    # Chuyển ma trận thành chuỗi dễ đọc
                    matrix_str = '\n'.join(['\t'.join(map(str, row)) for row in rounded_matrix])

                    # Hiển thị ma trận đã làm tròn
                    self.calibration_matrix_label.config(text=f"Perspective transformation matrix:\n{matrix_str}")
                    self.last_update_label.config(text=f"Offset X: {self.basler_camera.offset_x_cam_rb} mm\nOffset Y: {self.basler_camera.offset_y_cam_rb} mm\n\nLast update: {timestamp}")
        except Exception as e:
            self.update_calibrate_response(f"Error loading calibration data: {str(e)}")

    def toggle_calibrate_mode(self):
        """Toggle calibrate mode on/off"""
        if not self.calibrate_running:
            # Start calibrate mode
            self.calibrate_running = True
            self.calibrate_btn.config(text="STOP")
            self.update_btn.config(state='disabled')
            self.clicked_points = []  # Reset clicked points

            # Tắt camera ở các mode khác nếu đang bật
            if self.camera_state:
                self.toggle_camera()

            # Start camera và hiển thị hướng dẫn
            if not self.basler_camera.grabbing:
                if self.basler_camera.start_camera():
                    self.update_calibrate_response("Calibration Mode Started")
                    self.update_calibrate_response("\nClick points in order:")
                    self.update_calibrate_response("1. (-80,-80)  2. (0,-80)   3. (80,-80)")
                    self.update_calibrate_response("4. (-80,0)    5. (0,0)     6. (80,0)")
                    self.update_calibrate_response("7. (-80,80)   8. (0,80)    9. (80,80)")
                    self.start_calibrate_preview()
        else:
            # Stop calibrate mode
            self.stop_calibrate_mode()

    def stop_calibrate_mode(self):
        """Stop calibrate mode"""
        self.calibrate_running = False
        self.calibrate_btn.config(text="START")
        self.update_btn.config(state='disabled')
        self.clicked_points = []

        # Xóa binding chuột
        self.camera_label.unbind('<Motion>')
        self.camera_label.unbind('<Button-1>')
        self.camera_label.unbind('<Button-3>')
        self.camera_label.unbind('<Enter>')
        self.camera_label.unbind('<Leave>')

        # Reset cursor
        self.camera_label.configure(cursor="")

        # Xóa vị trí chuột cuối
        if hasattr(self, 'last_mouse_pos'):
            delattr(self, 'last_mouse_pos')

        # Tắt camera
        if self.basler_camera.grabbing:
            self.basler_camera.stop_camera()
            self.camera_label.configure(image='')
            self.camera_label.image = None

    def start_calibrate_preview(self):
        """Preview camera cho calibration"""
        if not self.calibrate_running:
            return

        frame = self.basler_camera.get_frame()
        if frame is not None:
            if self.basler_camera.camera.PixelFormat.GetValue() == "BayerRG8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)

            # Vẽ các điểm đã click
            for i, (point, real_coord) in enumerate(zip(self.clicked_points, self.real_points)):
                cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{i+1}:({real_coord[0]},{real_coord[1]})",
                          (point[0]+10, point[1]+10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Vẽ con trỏ chuột dạng dấu cộng màu đỏ
            if hasattr(self, 'last_mouse_pos'):
                x, y = self.last_mouse_pos
                # Vẽ đường ngang
                cv2.line(frame, (x-10, y), (x+10, y), (0, 0, 255), 1)  # Đổi (0, 255, 0) thành (0, 0, 255)
                # Vẽ đường dọc
                cv2.line(frame, (x, y-10), (x, y+10), (0, 0, 255), 1)  # Đổi (0, 255, 0) thành (0, 0, 255)

            # Hiển thị frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            frame_w = self.camera_frame.winfo_width()
            frame_h = self.camera_frame.winfo_height()
            if frame_w > 0 and frame_h > 0:
                scale = min(frame_w/w, frame_h/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo

            # Bind các event chuột và mouse enter/leave
            self.camera_label.bind('<Button-1>', self.on_calibrate_click)
            self.camera_label.bind('<Button-3>', self.on_calibrate_right_click)
            self.camera_label.bind('<Motion>', self.on_mouse_move)
            self.camera_label.bind('<Enter>', self.on_mouse_enter)
            self.camera_label.bind('<Leave>', self.on_mouse_leave)

        # Lập lịch cập nhật frame tiếp theo
        if self.calibrate_running:
            self.root.after(30, self.start_calibrate_preview)

    def on_calibrate_click(self, event):
        """Xử lý click chuột trong chế độ calibrate"""
        if len(self.clicked_points) < 9:
            # Tính toán tọa độ thực dựa trên kích thước hiển thị
            display_w = self.camera_label.winfo_width()
            display_h = self.camera_label.winfo_height()
            scale_x = self.basler_camera.WIDTH / display_w
            scale_y = self.basler_camera.HEIGHT / display_h

            real_x = int(event.x * scale_x)
            real_y = int(event.y * scale_y)

            self.clicked_points.append([real_x, real_y])
            self.update_calibrate_response(f"Point {len(self.clicked_points)} added at ({real_x}, {real_y})")

            if len(self.clicked_points) == 9:
                self.update_btn.config(state='normal')
                self.update_calibrate_response("\nAll points selected. Click 'Update' to calculate calibration matrix")

    def on_calibrate_right_click(self, event):
        """Xóa điểm cuối cùng khi click chuột phải"""
        if self.clicked_points:
            removed_point = self.clicked_points.pop()
            self.update_calibrate_response(f"Removed last point at ({removed_point[0]}, {removed_point[1]})")
            self.update_btn.config(state='disabled')

    def on_mouse_move(self, event):
        """Cập nhật vị trí chuột khi di chuyển"""
        if self.calibrate_running:
            display_w = self.camera_label.winfo_width()
            display_h = self.camera_label.winfo_height()
            scale_x = self.basler_camera.WIDTH / display_w
            scale_y = self.basler_camera.HEIGHT / display_h

            # Chuyển đổi tọa độ chuột sang tọa độ thực của frame
            real_x = int(event.x * scale_x)
            real_y = int(event.y * scale_y)
            self.last_mouse_pos = (real_x, real_y)

    def on_mouse_enter(self, event):
        """Ẩn con trỏ chuột khi vào vùng camera"""
        if self.calibrate_running:
            self.camera_label.configure(cursor=self.blank_cursor)

    def on_mouse_leave(self, event):
        """Hiện lại con trỏ chuột khi ra khỏi vùng camera"""
        if self.calibrate_running:
            self.camera_label.configure(cursor="")

    def update_calibration(self):
        """Cập nhật ma trận hiệu chuẩn"""
        if len(self.clicked_points) == 9:
            # Tính transform matrix
            img_points = np.array(self.clicked_points, dtype=np.float32)
            obj_points = np.array(self.real_points, dtype=np.float32)

            perspective_transform, mask = cv2.findHomography(img_points, obj_points)

            if np.all(perspective_transform == 0):
                self.update_calibrate_response("Error: Invalid transform matrix!")
                return

            # Lưu ma trận mới
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            transform_data = {
                'matrix': perspective_transform.tolist(),
                'timestamp': timestamp
            }

            try:
                with open(self.basler_camera.transform_file, 'w') as f:
                    json.dump(transform_data, f, indent=4)

                # Cập nhật hiển thị
                self.calibration_matrix_label.config(text=f"Calibration Matrix:\n{json.dumps(perspective_transform.tolist(), indent=2)}")
                self.last_update_label.config(text=f"Last update: {timestamp}")

                # Cập nhật ma trận trong camera object
                self.basler_camera.perspective_transform = perspective_transform

                self.update_calibrate_response("\nCalibration matrix updated successfully!")

                # Stop calibrate mode
                self.stop_calibrate_mode()

            except Exception as e:
                self.update_calibrate_response(f"Error saving calibration matrix: {str(e)}")

#----------------------------------------------------------------------------------#
#----------------------------- Print on command window ----------------------------#
#----------------------------------------------------------------------------------#
    def update_response(self, text, clear=False):
        """Command window in ALL MODE"""
        self.response_text.insert(tk.END, text + "\n")
        self.response_text.see(tk.END)
        self.auto_response_text.insert(tk.END, text + "\n")
        self.auto_response_text.see(tk.END)
        self.calibrate_response_text.insert(tk.END, text + "\n")
        self.calibrate_response_text.see(tk.END)
        if clear:
            self.response_text.delete(1.0, tk.END)
            self.auto_response_text.delete(1.0, tk.END)
            self.calibrate_response_text.delete(1.0, tk.END)

    def update_manual_response(self, text, clear=False):
        """Command window in MANUAL MODE"""
        self.response_text.insert(tk.END, text + "\n")
        self.response_text.see(tk.END)
        if clear:
            self.response_text.delete(1.0, tk.END)

    def update_auto_response(self, text, clear=False):
        """Command window in AUTO MODE"""
        self.auto_response_text.insert(tk.END, text + "\n")
        self.auto_response_text.see(tk.END)
        if clear:
            self.auto_response_text.delete(1.0, tk.END)

    def update_calibrate_response(self, text):
        """Command window in CALIB MODE"""
        self.calibrate_response_text.insert(tk.END, text + "\n")
        self.calibrate_response_text.see(tk.END)

#----------------------------------------------------------------------------------#
#----------------------------------- Pick and Place -------------------------------#
#----------------------------------------------------------------------------------#
    def execute_pick_and_place(self):
        """Thực hiện pick and place với tọa độ từ entry"""
        try:
            x = float(self.pp_x.get())
            y = float(self.pp_y.get())
            box = int(self.pp_box.get())  # Nhập số thứ tự của hộp (1 hoặc 2)

            if box not in [1, 2]:
                self.update_response("Invalid box number. Please enter 1 or 2.")
                return
            # Gọi hàm pick_n_place với tọa độ và số thứ tự hộp
            self.pick_n_place(x, y, box)

        except ValueError:
            self.update_response("Invalid coordinates. Please enter valid numbers.")
        except Exception as e:
            self.update_response(f"Error during pick and place: {str(e)}")

    def pick_n_place(self, x, y, no):
        """Thực hiện pick & place"""
        self.pick_in_progress = True
        self.suppress_response = True  # Bắt đầu ngăn phản hồi
        if x > -345:
            try:
                # pick & grip
                self.send_gcode(f"G01 X{x} Y{y} Z{z_grip + move_up} F{velo} A{acce}")
                self.send_gcode(f"G01 X{x} Y{y} Z{z_grip} F800 A1500")
                self.send_gcode("M03 D1")
                time.sleep(0.2)
                # move up
                self.send_gcode(f"G01 X{x} Y{y} Z{z_grip + move_up} F{velo} A{acce}")
                # place
                if no == 1: self.send_gcode(f"G01 X{x_re1} Y{y_re1} Z{z_re1} F{velo} A{acce}")
                elif no == 2: self.send_gcode(f"G01 X{x_re2} Y{y_re2} Z{z_re2} F{velo} A{acce}")
                # ungrip & home
                self.send_gcode("M05 D1")
                time.sleep(0.2)
                # self.send_gcode(f"G01 X0 Y0 Z{z_grip + move_up} F{velo} A{acce}")
                self.update_response(f"Picked at ({x: .1f}, {y: .1f}) to Receiver {no}")
            finally:
                self.suppress_response = False  # Kết thúc ngăn phản hồi
                self.pick_in_progress = False  # End pick and place

if __name__ == "__main__":
    os.system('cls')
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()
