import cv2
import numpy as np

def horn_schunck_optical_flow(img1, img2, alpha=1.0, num_iters=100):
    # Chuyển ảnh sang float64 để tránh tràn số khi tính đạo hàm
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    
    # Định nghĩa các bộ lọc lấy đạo hàm không gian và thời gian
    kx = np.array([[-1, 1], [-1, 1]]) * 0.25
    ky = np.array([[-1, -1], [1, 1]]) * 0.25
    kt = np.array([[1, 1], [1, 1]]) * 0.25
    
    Ix = cv2.filter2D(im1, -1, kx) + cv2.filter2D(im2, -1, kx)
    Iy = cv2.filter2D(im1, -1, ky) + cv2.filter2D(im2, -1, ky)
    It = cv2.filter2D(im2, -1, kt) - cv2.filter2D(im1, -1, kt)
    
    # Khởi tạo ma trận trường vận tốc ban đầu
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    
    # Ma trận tính trung bình 4 lân cận (Laplacian)
    k_avg = np.array([[0, 0.25, 0],
                      [0.25, 0, 0.25],
                      [0, 0.25, 0]])
    
    # Mẫu số chung cố định trong công thức cập nhật
    div = alpha**2 + Ix**2 + Iy**2
    
    # Vòng lặp Gauss-Seidel tìm nghiệm tối ưu
    for _ in range(num_iters):
        u_avg = cv2.filter2D(u, -1, k_avg)
        v_avg = cv2.filter2D(v, -1, k_avg)
        
        # Tính toán sai số độ sáng (brightness constraint error)
        err = (Ix * u_avg + Iy * v_avg + It) / div
        
        # Cập nhật lại dòng quang học
        u = u_avg - Ix * err
        v = v_avg - Iy * err
        
    return u, v

def draw_quiver(img, u, v, step=12):
    h, w = img.shape
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = u[y, x], v[y, x]
            
            # Chỉ vẽ mũi tên khi độ lớn dịch chuyển vượt ngưỡng nhiễu
            if np.sqrt(fx**2 + fy**2) > 2.5: 
                p1 = (x, y)
                p2 = (int(x + fx * 3), int(y + fy * 3)) # Nhân 3 để phóng đại vector cho dễ nhìn
                
                cv2.arrowedLine(out, p1, p2, (0, 255, 0), 1, tipLength=0.3)
                
    return out