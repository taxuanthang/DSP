import cv2
import numpy as np

# --- BỘ LỌC THÔNG THẤP LÝ TƯỞNG (IDEAL LOWPASS) ---
def ideal_lowpass_filter(img, radius=50):
    src = img.astype(np.float64)
    h, w = src.shape
    
    # FFT 2D và dịch tâm phổ
    f_shift = np.fft.fftshift(np.fft.fft2(src))
    
    # Tính khoảng cách từ tâm d để làm mặt nạ
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    # Tạo filter Ideal cắt thẳng tần số cao ngoài bán kính
    mask = np.zeros((h, w))
    mask[d <= radius] = 1.0
    
    # Nhân ma trận lọc trong miền tần số
    f_out = f_shift * mask
    
    # IFFT đưa về miền không gian ảnh
    img_back = np.fft.ifft2(np.fft.ifftshift(f_out))
    res = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
    
    # Tính phổ biên độ dạng log để hiển thị lên đồ thị GUI
    spec_orig = np.log(1 + np.abs(f_shift))
    spec_orig = cv2.normalize(spec_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    spec_filt = np.log(1 + np.abs(f_out))
    spec_filt = cv2.normalize(spec_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask

# --- BỘ LỌC THÔNG CAO LÝ TƯỞNG (IDEAL HIGHPASS) ---
def ideal_highpass_filter(img, radius=10):
    src = img.astype(np.float64)
    h, w = src.shape
    
    # Biến đổi Fourier sang miền tần số
    f_shift = np.fft.fftshift(np.fft.fft2(src))
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    # Ngược lại với lowpass: Chặn vùng bên trong, giữ bên ngoài
    mask = np.ones((h, w))
    mask[d <= radius] = 0.0
    
    f_out = f_shift * mask
    
    # Khôi phục ảnh ngược
    img_back = np.fft.ifft2(np.fft.ifftshift(f_out))
    res = cv2.normalize(np.abs(img_back), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Map phổ log tần số
    spec_orig = cv2.normalize(np.log(1 + np.abs(f_shift)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_filt = cv2.normalize(np.log(1 + np.abs(f_out)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask

# --- BỘ LỌC THÔNG THẤP BUTTERWORTH ---
def butterworth_lowpass_filter(img, radius=50, n=2):
    src = img.astype(np.float64)
    h, w = src.shape
    
    f_shift = np.fft.fftshift(np.fft.fft2(src))
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    # Công thức Butterworth hạ bậc mịn màng thay vì cắt cụt
    mask = 1.0 / (1.0 + (d / radius) ** (2 * n))
    
    f_out = f_shift * mask
    
    img_back = np.fft.ifft2(np.fft.ifftshift(f_out))
    res = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
    
    spec_orig = cv2.normalize(np.log(1 + np.abs(f_shift)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_filt = cv2.normalize(np.log(1 + np.abs(f_out)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask

# --- BỘ LỌC THÔNG CAO GAUSSIAN ---
def gaussian_highpass_filter(img, radius=10):
    src = img.astype(np.float64)
    h, w = src.shape
    
    f_shift = np.fft.fftshift(np.fft.fft2(src))
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    # Công thức phân bố Gauss ngược lọc lấy biên cạnh sắc
    mask = 1.0 - np.exp(-(d ** 2) / (2 * (radius ** 2)))
    
    f_out = f_shift * mask
    
    img_back = np.fft.ifft2(np.fft.ifftshift(f_out))
    res = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
    
    spec_orig = cv2.normalize(np.log(1 + np.abs(f_shift)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_filt = cv2.normalize(np.log(1 + np.abs(f_out)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask

# --- BỘ LỌC CHẶN DẢI NOTCH (NOTCH REJECT FILTER) ---
def notch_reject_filter(img, uk, vk, radius=10):
    src = img.astype(np.float64)
    h, w = src.shape
    
    f_shift = np.fft.fftshift(np.fft.fft2(src))
    mask = np.ones((h, w), dtype=np.float64)
    cy, cx = h // 2, w // 2
    
    # Hàm con đục lỗ triệt tiêu nhiễu sọc tuần hoàn
    def clear_noise_spot(m_matrix, py, px, r):
        Y, X = np.ogrid[:m_matrix.shape[0], :m_matrix.shape[1]]
        dist = np.sqrt((Y - py)**2 + (X - px)**2)
        m_matrix[dist <= r] = 0
        return m_matrix

    # Nếu click chuột chọn tọa độ nhiễu trên giao diện
    if uk != 30 or vk != 30:
        mask = clear_noise_spot(mask, cy + vk, cx + uk, r=radius)
        mask = clear_noise_spot(mask, cy - vk, cx - uk, r=radius)
    else:
        # Nếu chạy mặc định ban đầu, tự động tính sọc chu kỳ theo đề bài (T = 20)
        period = 20
        offset = int(w / period)
        mask = clear_noise_spot(mask, cy, cx - offset, r=radius)
        mask = clear_noise_spot(mask, cy, cx + offset, r=radius)
        
    f_notch = f_shift * mask
    
    # Phục hồi ảnh sạch nhiễu
    img_back = np.fft.ifft2(np.fft.ifftshift(f_notch))
    res = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
    
    return res, None, f_notch, None