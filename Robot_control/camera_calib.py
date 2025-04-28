import cv2
import numpy as np
import json
import time
from pypylon import pylon

def main():
    # ----------------- CÀI ĐẶT CHECKERBOARD -----------------
    squares_x = 8        # Số ô theo chiều ngang
    squares_y = 6        # Số ô theo chiều dọc
    square_length = 25   # Độ dài mỗi ô (mm)

    # Tạo các tham số phát hiện checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ----------------- MỞ BASLER CAMERA -----------------
    try:
        # Tạo đối tượng camera từ pypylon
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        camera.PixelFormat.SetValue("BayerRG8")
        camera.Width.SetValue(1280)
        camera.Height.SetValue(1024)
        # Bắt đầu grab frame với chiến lược lấy ảnh mới nhất
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    except Exception as e:
        print("Lỗi mở camera:", str(e))
        return

    # Cấu hình converter để chuyển frame sang định dạng BGR8 (để dùng OpenCV)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # ----------------- HƯỚNG DẪN HIỆU CHUẨN -----------------
    print("Quy trình hiệu chuẩn:")
    print("1. Đặt checkerboard vào vùng nhìn của camera.")
    print("2. Điều chỉnh board sao cho có số lượng góc đủ (>=20 góc được phát hiện).")
    print("3. Khi board ở vị trí hợp lệ (hiển thị 'Valid'), nhấn Enter để chụp ảnh.")
    print("4. Chụp tổng cộng 20 ảnh với các góc nhìn khác nhau.")
    print("5. Nhấn 'q' để thoát quá trình bất cứ lúc nào.\n")

    collected_frames = 0
    all_corners = []  # Danh sách chứa các góc checkerboard từng ảnh
    object_points = []  # Danh sách chứa các điểm 3D cho checkerboard
    image_size = None

    # Tạo các điểm 3D cho checkerboard
    objp = np.zeros((squares_y * squares_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2) * square_length

    # ----------------- VÒNG LẶP CHỤP ẢNH HIỆU CHUẨN -----------------
    while collected_frames < 20:
        try:
            result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        except Exception as e:
            print("Lỗi RetrieveResult:", str(e))
            break

        if result.GrabSucceeded():
            image = converter.Convert(result)
            frame = image.GetArray()
        else:
            print("Lỗi: Không grab được frame")
            result.Release()
            continue

        result.Release()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)

        valid = False
        if ret:
            # Tinh chỉnh các góc
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            frame_markers = cv2.drawChessboardCorners(frame.copy(), (squares_x, squares_y), corners2, ret)
            valid = True  # Nếu tìm thấy góc, ảnh hợp lệ
        else:
            frame_markers = frame.copy()
            cv2.putText(frame_markers, "No checkerboard detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame_markers, f"Captured frames: {collected_frames}/20", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Calibration", frame_markers)
        key = cv2.waitKey(1) & 0xFF

        # Nhấn Enter để lưu ảnh nếu ảnh hợp lệ
        if key == 13 or key == 10:
            if valid:
                if image_size is None:
                    image_size = gray.shape[::-1]  # (width, height)
                all_corners.append(corners2)
                object_points.append(objp)  # Thêm các điểm 3D vào danh sách
                collected_frames += 1
                print(f"Đã chụp ảnh {collected_frames}/20.")
            else:
                print("Ảnh không hợp lệ cho hiệu chuẩn. Điều chỉnh board và thử lại.")
        if key == ord('q'):
            break

    # Dừng grab và đóng camera
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

    if collected_frames < 4:
        print("Không đủ ảnh hợp lệ cho hiệu chuẩn. Kết thúc chương trình.")
        return

    # ----------------- HIỆU CHUẨN CAMERA -----------------
    print("Đang hiệu chuẩn camera với các ảnh thu thập được...")
    ret_val, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, all_corners, image_size, None, None)  # Cập nhật tham số

    # Tính toán ma trận chuyển đổi phối cảnh
    h, w = image_size
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

    print("\nHiệu chuẩn thành công!")
    print("Sai số reprojection:", ret_val)
    print("Ma trận camera (cameraMatrix):")
    print(cameraMatrix)
    print("Hệ số méo (distCoeffs):")
    print(distCoeffs)

    # ----------------- LƯU THÔNG SỐ HIỆU CHUẨN -----------------
    calib_data = {
        "camera_matrix": cameraMatrix.tolist(),
        "dist_coeffs": distCoeffs.tolist(),
        "reprojection_error": ret_val,
        "new_camera_matrix": new_camera_matrix.tolist(),  # Thêm ma trận mới
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("calibration_parameters.json", "w") as f:
        json.dump(calib_data, f, indent=4)
    print("\nThông số hiệu chuẩn đã được lưu vào file 'calibration_parameters.json'.")

if __name__ == '__main__':
    main()