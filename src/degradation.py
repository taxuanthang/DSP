import cv2
import numpy as np
import skimage.metrics as metrics

# Hàm tạo nhiễu và làm nhòe chuyển động nằm ngang
def apply_motion_blur(img, length=30, noise_sigma=5.0):
    src = img.astype(np.float64)
    h, w = src.shape
    
    # Khởi tạo ma trận ô lọc PSF (Point Spread Function)
    psf = np.zeros((h, w))
    cy, cx = h // 2, w // 2
    x1 = cx - length // 2
    x2 = x1 + length
    psf[cy, x1:x2] = 1.0 / length
    
    # Biến đổi Fourier và khử dịch pha bằng ifftshift
    F_img = np.fft.fft2(src)
    H = np.fft.fft2(np.fft.ifftshift(psf))
    
    # Nhân chập trên miền tần số
    F_blur = F_img * H
    img_blur = np.real(np.fft.ifft2(F_blur))
    
    # Cộng nhiễu Gaussian nếu có cấu hình sigma
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img_blur.shape)
        img_noisy = img_blur + noise
    else:
        img_noisy = img_blur
        
    # Ép kiểu uint8 trả về hiển thị giao diện
    img_uint8 = np.clip(img_noisy, 0, 255).astype(np.uint8)
    return img_noisy, img_uint8, H

# Hàm mô phỏng nhòe mờ do mất nét (Defocus Blur)
def apply_defocus_blur(img, radius=15, noise_sigma=5.0):
    src = img.astype(np.float64)
    h, w = src.shape
    
    # Tạo mặt nạ hình tròn làm bộ lọc nhòe
    psf = np.zeros((h, w))
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_sq = (Y - cy)**2 + (X - cx)**2
    psf[dist_sq <= radius**2] = 1.0
    psf /= np.sum(psf) # Chuẩn hóa năng lượng bộ lọc
    
    # Chuyển đổi hệ thống sang miền tần số
    F_img = np.fft.fft2(src)
    H = np.fft.fft2(np.fft.ifftshift(psf))
    
    # Tạo ảnh lỗi
    img_blur = np.real(np.fft.ifft2(F_img * H))
    if noise_sigma > 0:
        img_noisy = img_blur + np.random.normal(0, noise_sigma, img_blur.shape)
    else:
        img_noisy = img_blur
        
    img_uint8 = np.clip(img_noisy, 0, 255).astype(np.uint8)
    return img_noisy, img_uint8, H

# Thuật toán lọc phục hồi ảnh Wiener ngược chập kết hợp quét tối ưu SSIM
def wiener_deconvolution(img_degraded, F_psf, img_origin, K_candidates=None):
    if K_candidates is None:
        K_candidates = [0.001, 0.01, 0.05, 0.1, 0.5]
        
    F_deg = np.fft.fft2(img_degraded.astype(np.float64))
    H_conj = np.conj(F_psf)
    H_mag2 = np.abs(F_psf) ** 2
    
    best_ssim = -1.0
    best_k = 0.01
    best_psnr = -1
    best_snr = -1
    res_img = None
    
    ref_uint8 = np.clip(img_origin, 0, 255).astype(np.uint8)
    
    # Vòng lặp tìm kiếm hằng số K tối ưu nhất
    for k in K_candidates:
        # Công thức toán học Wiener nghịch đảo
        W = H_conj / (H_mag2 + k)
        F_hat = F_deg * W
        
        # Đưa ngược lại miền không gian phẳng
        img_back = np.abs(np.fft.ifft2(F_hat))
        img_uint8 = np.clip(img_back, 0, 255).astype(np.uint8)
        
        # Đánh giá độ tương đồng cấu trúc ảnh bằng SSIM
        score = metrics.structural_similarity(ref_uint8, img_uint8, data_range=255)
        current_ssim = metrics.structural_similarity(
            ref_uint8,
            img_uint8,
            data_range=255
        )

        current_psnr = compute_psnr(
            ref_uint8,
            img_uint8
        )

        current_snr = compute_snr(
            ref_uint8,
            img_uint8
        )

        print(
            f"K={k:<7}"
            f" SSIM={current_ssim:.4f}"
            f" PSNR={current_psnr:.2f}dB"
            f" SNR={current_snr:.2f}dB"
        )
        if current_ssim > best_ssim:

            best_ssim = current_ssim
            best_psnr = current_psnr
            best_snr = current_snr

            best_K = k
            best_img_restored = img_uint8
        return (
            best_img_restored,
            best_K,
            best_ssim,
            best_psnr,
            best_snr
        )
            
    return res_img, best_k, max_ssim

def compute_psnr(img_ref, img_test):
    """
    Peak Signal-to-Noise Ratio
    """

    img_ref = img_ref.astype(np.float64)
    img_test = img_test.astype(np.float64)

    mse = np.mean(
        (img_ref - img_test) ** 2
    )

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10(
        (255 ** 2) / mse
    )

    return psnr

def compute_snr(img_ref, img_test):
    """
    Signal-to-Noise Ratio
    """

    img_ref = img_ref.astype(np.float64)
    img_test = img_test.astype(np.float64)

    signal_power = np.mean(
        img_ref ** 2
    )

    noise_power = np.mean(
        (img_ref - img_test) ** 2
    )

    if noise_power == 0:
        return float("inf")

    snr = 10 * np.log10(
        signal_power / noise_power
    )

    return snr

# Hàm trích xuất đặc trưng hướng vân bề mặt dựa trên phổ năng lượng
def extract_texture_features(img, num_angles=8):
    # Tính phổ năng lượng 2D Power Spectrum
    F_shift = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    p_spectrum = np.abs(F_shift) ** 2
    
    h, w = p_spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[-cy : h - cy, -cx : w - cx]
    
    # Tính ma trận góc radian cục bộ của từng tần số
    rad_angles = np.arctan2(Y, X)
    feats = np.zeros(num_angles)
    step = 2 * np.pi / num_angles
    
    # Chia góc chạy lũy tiến quét quanh vòng tròn 360 độ
    for idx in range(num_angles):
        low_bound = -np.pi + idx * step
        high_bound = low_bound + step
        
        # Lọc dải góc tần số mong muốn
        zone_mask = (rad_angles >= low_bound) & (rad_angles < high_bound)
        zone_mask[cy, cx] = False # Loại bỏ điểm tần số trung tâm DC
        
        # Tích phân rời rạc (tổng) năng lượng phân bố
        feats[idx] = np.sum(p_spectrum[zone_mask])
        
    # Chuẩn hóa ma trận vector đặc trưng đầu ra
    s_val = np.sum(feats)
    if s_val > 0:
        feats /= s_val
        
    return feats