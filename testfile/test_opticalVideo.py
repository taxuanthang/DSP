import cv2
import numpy as np
import os
# Import thuật toán Horn-Schunck từ module bạn đã đóng gói
from src.opticalFlowEstimation import horn_schunck_optical_flow, visualize_flow_quiver

def process_video_optical_flow(video_path=0, output_path="results/output_optical_flow.mp4"):
    """
    Đọc video (hoặc webcam), áp dụng Horn-Schunck Optical Flow theo thời gian thực
    và lưu lại thành file video kết quả.
    
    Parameters:
    -----------
    video_path : str hoặc int
        Đường dẫn file video (.mp4, .avi). Nếu để bằng 0 sẽ tự động bật Webcam máy tính.
    output_path : str
        Đường dẫn lưu video kết quả sau khi đã vẽ mũi tên chuyển động.
    """
    # 1. Khởi tạo đối tượng đọc Video của OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở nguồn video hoặc webcam: {video_path}")
        return

    # Đọc khung hình đầu tiên để lấy kích thước
    ret, frame1 = cap.read()
    if not ret:
        print("[LỖI] Video trống hoặc không thể đọc khung hình đầu tiên.")
        cap.release()
        return

    # Chuyển khung hình đầu tiên sang ảnh xám
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Để thuật toán chạy mượt và nhanh, ta nên giảm bớt kích thước khung hình nếu video quá nặng
    # (Ví dụ: scale về chiều rộng 400 pixel)
    h, w = prev_gray.shape
    scale_w = 400
    scale_h = int(h * (scale_w / w))
    prev_gray = cv2.resize(prev_gray, (scale_w, scale_h))

    # Cấu hình đối tượng Ghi Video (VideoWriter) để xuất file báo cáo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, 15.0, (scale_w, scale_h))

    print("\n--- BẮT ĐẦU PHÂN TÍCH CHUYỂN ĐỘNG VIDEO (Ấn 'q' để thoát) ---")
    
    frame_count = 0
    while True:
        # Đọc khung hình tiếp theo
        ret, frame2 = cap.read()
        if not ret:
            print("Đã xử lý hết video hoặc luồng video bị ngắt.")
            break
            
        frame_count += 1
        # Tiền xử lý khung hình tiếp theo (Xám + Resize đồng bộ)
        curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (scale_w, scale_h))
        
        # Làm mịn cả 2 khung hình bằng bộ lọc Gaussian cỡ hạt nhân (5,5)
        prev_gray_smoothed = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_gray_smoothed = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        # 2. Thực thi Horn-Schunck tính vector dịch chuyển (u, v) giữa 2 khung liên tiếp
        # Giảm số vòng lặp num_iters xuống khoảng 20-30 để video chạy mượt mà, không bị khựng
        u, v = horn_schunck_optical_flow(prev_gray, curr_gray, alpha=10.0, num_iters=100)
        
        # 3. Trực quan hóa: Vẽ các mũi tên hướng chuyển động lên ảnh kết quả
        # step=10 hoặc 12 nghĩa là cứ cách 12 pixel vẽ 1 mũi tên để tránh rối mắt
        result_frame = visualize_flow_quiver(curr_gray, u, v, step=30)
        
        # Ghi khung hình đã xử lý vào file video đầu ra
        out_video.write(result_frame)
        
        # Hiển thị trực tiếp tiến trình lên màn hình GUI của OpenCV
        cv2.imshow("Horn-Schunck Optical Flow - Video Analysis", result_frame)
        
        # Gối đầu: Khung hình hiện tại trở thành khung hình cũ cho lượt lặp kế tiếp
        prev_gray = curr_gray
        
        # Nhấn phím 'q' trên bàn phím để dừng video chủ động
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Người dùng chủ động dừng tiến trình.")
            break

    # Thu dọn tài nguyên hệ thống
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"--- HOÀN THÀNH --- Video kết quả được lưu tại: {output_path}")

if __name__ == "__main__":
    # KỊCH BẢN CHẠY:
    # Cách A: Truyền đường dẫn file video thực tế của nhóm bạn vào đây:
    video_source = "data/testVideo5.mp4"
    
    # Cách B: Để bằng số 0 để thuật toán tự động bật WEBCAM của laptop bạn lên test chuyển động cơ thể:
    # video_source = 0 
    
    process_video_optical_flow(video_path=video_source)