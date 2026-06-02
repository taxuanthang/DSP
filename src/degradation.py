import cv2
import numpy as np
import skimage.metrics as metrics

def apply_motion_blur(img, length=30, noise_sigma=5.0):
    """
    Mô phỏng hiệu ứng nhòe chuyển động nằm ngang (Motion Blur) và cộng nhiễu Gaussian.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Ảnh gốc sạch đầu vào (2D array).
    length : int
        Độ dài vệt nhòe chuyển động (L = 10, 30, 50).
    noise_sigma : float
        Độ lệch chuẩn của nhiễu trắng Gaussian cộng thêm vào ảnh.
        
    Returns:
    --------
    img_degraded : numpy.ndarray (float64)
        Mảng ảnh bị lỗi (chưa ép kiểu uint8) phục vụ cho tính toán bộ lọc Wiener.
    img_degraded_uint8 : numpy.ndarray (uint8)
        Ảnh bị lỗi định dạng chuẩn 0-255 để lưu file hoặc hiển thị GUI.
    F_psf : numpy.ndarray
        Phổ tần số Fourier 2D của ma trận làm nhòe PSF (H(u,v)), dùng cho lọc Wiener.
    """
    img_float = img.astype(np.float64)
    rows, cols = img_float.shape
    
    # 1. Tạo ma trận điểm lan truyền PSF (Point Spread Function) nằm ngang
    psf = np.zeros((rows, cols))
    center_y = rows // 2
    center_x = cols // 2
    start_x = center_x - length // 2
    end_x = start_x + length
    psf[center_y, start_x:end_x] = 1.0 / length
    
    # 2. Chuyển đổi sang miền tần số (áp dụng ifftshift để tránh dịch pha lệch hình)
    F_img = np.fft.fft2(img_float)
    F_psf = np.fft.fft2(np.fft.ifftshift(psf))
    
    # 3. Nhân chập miền tần số và biến đổi ngược để tạo ảnh nhòe
    F_blurred = F_img * F_psf
    img_blurred = np.abs(np.fft.ifft2(F_blurred))
    
    # 4. Cộng nhiễu trắng Gaussian (AWGN)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img_blurred.shape)
        img_degraded = img_blurred + noise
    else:
        img_degraded = img_blurred
        
    # Ép kiểu dải giá trị chuẩn hiển thị và lưu trữ 
    img_degraded_uint8 = np.clip(img_degraded, 0, 255).astype(np.uint8)
    
    return img_degraded, img_degraded_uint8, F_psf

def wiener_deconvolution(img_degraded, F_psf, img_origin, K_candidates=None):
    """
    Khôi phục ảnh bị nhòe bằng bộ lọc Wiener và tối ưu hóa tham số K tự động bằng SSIM.
    
    Parameters:
    -----------
    img_degraded : numpy.ndarray
        Ảnh bị nhòe/nhiễu đầu vào (2D array).
    F_psf : numpy.ndarray
        Phổ tần số Fourier 2D của hàm làm nhòe thu được từ Mục 9.
    img_origin : numpy.ndarray
        Ảnh gốc sạch ban đầu để tính toán chỉ số SSIM làm căn cứ tối ưu.
    K_candidates : list, optional
        Danh sách các giá trị hằng số K để quét thử nghiệm.
        
    Returns:
    --------
    best_img_restored : numpy.ndarray (uint8)
        Ảnh khôi phục cho kết quả SSIM cao nhất.
    best_K : float
        Giá trị K tối ưu nhất được tìm thấy.
    best_ssim : float
        Giá trị SSIM cao nhất đạt được.
    """
    if K_candidates == None:
        K_candidates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
        
    # Tính phổ Fourier 2D của ảnh lỗi đầu vào (Không dịch tâm để nhân trực tiếp với F_psf)
    F_degraded = np.fft.fft2(img_degraded.astype(np.float64))
    
    # Tính các thành phần cố định trong công thức Wiener để tối ưu tốc độ
    psf_fft_conj = np.conj(F_psf)
    psf_fft_mag2 = np.abs(F_psf) ** 2
    
    best_ssim = -1
    best_K = None
    best_img_restored = None
    
    img_origin_uint8 = np.clip(img_origin, 0, 255).astype(np.uint8)
    
    # Quét qua danh sách tham số K
    for K in K_candidates:
        # Áp dụng công thức bộ lọc Wiener W(u,v)
        W = psf_fft_conj / (psf_fft_mag2 + K)
        F_hat = F_degraded * W
        
        # Biến đổi ngược về miền không gian
        img_restored = np.abs(np.fft.ifft2(F_hat))
        img_restored_uint8 = np.clip(img_restored, 0, 255).astype(np.uint8)
        
        # Tính toán SSIM kiểm tra chất lượng khôi phục
        current_ssim = metrics.structural_similarity(img_origin_uint8, img_restored_uint8, data_range=255)
        print(f"[Wiener Scan] Thử nghiệm K = {K:<7} -> SSIM đạt: {current_ssim:.4f}")
        
        # Giữ lại cấu hình xuất sắc nhất
        if current_ssim > best_ssim:
            best_ssim = current_ssim
            best_K = K
            best_img_restored = img_restored_uint8
            
    return best_img_restored, best_K, best_ssim

def extract_texture_features(img, num_angles=8):
    """
    Trích xuất đặc trưng năng lượng theo hướng (angular bins) từ phổ tần số của ảnh.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Ảnh xám đầu vào (2D array).
    num_angles : int
        Số lượng ô hướng chia trên vòng tròn (mặc định là 8).
        
    Returns:
    --------
    feature_vector : numpy.ndarray
        Vector đặc trưng đã chuẩn hóa (tổng bằng 1), độ dài bằng num_angles.
    """
    # 1. Biến đổi sang miền tần số
    img_float = img.astype(np.float64)
    F = np.fft.fft2(img_float)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    
    # 2. Tạo ma trận góc tọa độ từ tâm phổ
    rows, cols = mag.shape
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    
    # Tính góc của từng pixel (từ -pi đến pi)
    angles = np.arctan2(Y, X)
    
    feature_vector = []
    angle_step = 2 * np.pi / num_angles
    
    for i in range(num_angles):
        start_angle = -np.pi + i * angle_step
        end_angle = start_angle + angle_step
        
        # Tạo mặt nạ chọn các pixel nằm trong dải góc này
        mask = (angles >= start_angle) & (angles < end_angle)
        
        # Bỏ qua điểm tâm DC (tần số bằng 0) để không làm nhiễu tổng năng lượng
        mask[cy, cx] = False 
        
        # Tính tổng năng lượng (biên độ) trong ô hướng này
        energy = np.sum(mag[mask])
        feature_vector.append(energy)
        
    # Chuẩn hóa vector đặc trưng về tổng bằng 1
    feature_vector = np.array(feature_vector)
    total_energy = np.sum(feature_vector)
    if total_energy > 0:
        feature_vector /= total_energy
        
    return feature_vector

