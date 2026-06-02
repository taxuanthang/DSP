import cv2
import numpy as np

def ideal_lowpass_filter(img, radius=50):
    """
    Áp dụng bộ lọc Ideal Lowpass Filter (ILPF) lên ảnh trong miền tần số.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Ảnh xám đầu vào (2D array).
    radius : int
        Tần số cắt D0 (bán kính vùng giữ lại xung quanh tâm).
        
    Returns:
    --------
    img_filtered : numpy.ndarray (uint8)
        Ảnh sau khi lọc sạch tần số cao.
    spectrum_orig : numpy.ndarray
        Phổ biên độ log của ảnh gốc (để hiển thị GUI).
    spectrum_filt : numpy.ndarray
        Phổ biên độ log của ảnh sau khi lọc (để hiển thị GUI).
    mask : numpy.ndarray
        Mặt nạ lọc dạng nhị phân (0 và 1).
    """
    # 1. Chuyển sang float để tính toán chính xác
    img_float = img.astype(np.float64)
    rows, cols = img_float.shape
    
    # 2. Biến đổi Fourier 2D và dịch tâm
    F = np.fft.fft2(img_float)
    F_shift = np.fft.fftshift(F)
    
    # 3. Tạo mặt nạ lọc tối ưu hóa bằng NumPy (thay cho vòng lặp for)
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    
    mask = np.zeros((rows, cols))
    mask[distance <= radius] = 1.0
    
    # 4. Áp dụng mặt nạ lọc
    F_filtered_shift = F_shift * mask
    
    # 5. Biến đổi Fourier ngược để lấy lại ảnh miền không gian
    F_inverse_shift = np.fft.ifftshift(F_filtered_shift)
    img_back = np.fft.ifft2(F_inverse_shift)
    img_filtered = np.abs(img_back)
    
    # Ép kiểu dải giá trị về chuẩn uint8 (0-255) trước khi trả về
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    # 6. Tính toán phổ biên độ log (Được chuẩn hóa để hiển thị đẹp trên GUI)
    spectrum_orig = np.log(1 + np.abs(F_shift))
    spectrum_orig = cv2.normalize(spectrum_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    spectrum_filt = np.log(1 + np.abs(F_filtered_shift))
    spectrum_filt = cv2.normalize(spectrum_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_filtered, spectrum_orig, spectrum_filt, mask

def ideal_highpass_filter(img, radius=10):
    """
    Áp dụng bộ lọc Ideal Highpass Filter (IHPF) lên ảnh trong miền tần số.
    Giữ lại các tần số cao (biên, cạnh) và loại bỏ tần số thấp.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Ảnh xám đầu vào (2D array).
    radius : int
        Tần số cắt D0 (bán kính vùng bị chặn xung quanh tâm).
        
    Returns:
    --------
    img_filtered : numpy.ndarray (uint8)
        Ảnh sau khi lọc thông cao (chỉ còn lại các đường biên/cạnh sắc).
    spectrum_orig : numpy.ndarray
        Phổ biên độ log của ảnh gốc.
    spectrum_filt : numpy.ndarray
        Phổ biên độ log của ảnh sau khi lọc.
    mask : numpy.ndarray
        Mặt nạ lọc thông cao nhị phân.
    """
    # 1. Chuyển sang float để tính toán chính xác
    img_float = img.astype(np.float64)
    rows, cols = img_float.shape
    
    # 2. Biến đổi Fourier 2D và dịch tâm
    F = np.fft.fft2(img_float)
    F_shift = np.fft.fftshift(F)
    
    # 3. Tạo mặt nạ lọc thông cao tối ưu bằng NumPy (1 - mask_low)
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    
    mask = np.ones((rows, cols))
    mask[distance <= radius] = 0.0  # Chặn các tần số thấp bên trong bán kính
    
    # 4. Áp dụng mặt nạ lọc lên phổ đã dịch tâm
    F_filtered_shift = F_shift * mask
    
    # 5. Biến đổi Fourier ngược
    F_inverse_shift = np.fft.ifftshift(F_filtered_shift)
    img_back = np.fft.ifft2(F_inverse_shift)
    img_filtered = np.abs(img_back)
    
    # 6. Chuẩn hóa ảnh đầu ra về [0, 255] chuẩn uint8 đúng như code gốc của bạn
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 7. Tính toán phổ biên độ log để hiển thị đồ thị / GUI
    spectrum_orig = np.log(1 + np.abs(F_shift))
    spectrum_orig = cv2.normalize(spectrum_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    spectrum_filt = np.log(1 + np.abs(F_filtered_shift))
    spectrum_filt = cv2.normalize(spectrum_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_filtered, spectrum_orig, spectrum_filt, mask

def butterworth_lowpass_filter(img, radius=50, n=2):
    """
    Áp dụng bộ lọc Butterworth Lowpass Filter (BLPF) lên ảnh trong miền tần số.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Ảnh xám đầu vào (2D array).
    radius : int/float
        Tần số cắt D0 (bán kính vùng giữ lại xung quanh tâm).
    n : int
        Bậc của bộ lọc (thường từ 1 đến 10). Bậc càng cao, cạnh cắt càng dốc.
        
    Returns:
    --------
    img_filtered : numpy.ndarray (uint8)
        Ảnh sau khi lọc thông thấp Butterworth (làm mịn ảnh).
    spectrum_orig : numpy.ndarray
        Phổ biên độ log của ảnh gốc.
    spectrum_filt : numpy.ndarray
        Phổ biên độ log của ảnh sau khi lọc.
    mask : numpy.ndarray
        Mặt nạ lọc Butterworth Lowpass (giá trị thực từ 0 đến 1).
    """
    # 1. Chuyển sang float để tính toán chính xác
    img_float = img.astype(np.float64)
    rows, cols = img_float.shape
    
    # 2. Biến đổi Fourier 2D và dịch tâm
    F = np.fft.fft2(img_float)
    F_shift = np.fft.fftshift(F)
    
    # 3. Tạo ma trận khoảng cách từ tâm ảnh
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    
    # 4. Công thức toán học hàm truyền đạt H(u,v) của Butterworth Lowpass
    mask = 1.0 / (1.0 + (distance / radius) ** (2 * n))
    
    # 5. Áp dụng mặt nạ lọc lên phổ đã dịch tâm
    F_filtered_shift = F_shift * mask
    
    # 6. Biến đổi Fourier ngược để lấy lại ảnh miền không gian
    F_inverse_shift = np.fft.ifftshift(F_filtered_shift)
    img_back = np.fft.ifft2(F_inverse_shift)
    img_filtered = np.abs(img_back)
    
    # Ép kiểu dải giá trị về chuẩn ảnh xám uint8 (0-255)
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    # 7. Tính toán phổ biên độ log để hiển thị đồ thị / GUI
    spectrum_orig = np.log(1 + np.abs(F_shift))
    spectrum_orig = cv2.normalize(spectrum_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    spectrum_filt = np.log(1 + np.abs(F_filtered_shift))
    spectrum_filt = cv2.normalize(spectrum_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_filtered, spectrum_orig, spectrum_filt, mask

def gaussian_highpass_filter(img, radius=10):
    """
    Áp dụng bộ lọc Gaussian Highpass Filter (GHPF) lên ảnh trong miền tần số.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Ảnh xám đầu vào (2D array).
    radius : int/float
        Tần số cắt D0 (độ rộng của hàm chuông Gauss ngược).
        
    Returns:
    --------
    img_filtered : numpy.ndarray (uint8)
        Ảnh sau khi lọc thông cao Gaussian (làm nổi bật đường biên, cạnh sắc).
    spectrum_orig : numpy.ndarray
        Phổ biên độ log của ảnh gốc.
    spectrum_filt : numpy.ndarray
        Phổ biên độ log của ảnh sau khi lọc.
    mask : numpy.ndarray
        Mặt nạ lọc Gaussian Highpass (vùng tâm bằng 0, càng xa tâm càng tiến về 1).
    """
    # 1. Chuyển sang float để tính toán chính xác
    img_float = img.astype(np.float64)
    rows, cols = img_float.shape
    
    # 2. Biến đổi Fourier 2D và dịch tâm
    F = np.fft.fft2(img_float)
    F_shift = np.fft.fftshift(F)
    
    # 3. Tạo ma trận khoảng cách từ tâm ảnh bằng ogrid giống code gốc của bạn
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    
    # 4. Công thức toán học hàm truyền đạt H(u,v) của Gaussian Highpass
    mask = 1.0 - np.exp(-(distance ** 2) / (2 * (radius ** 2)))
    
    # 5. Áp dụng mặt nạ lọc lên phổ đã dịch tâm
    F_filtered_shift = F_shift * mask
    
    # 6. Biến đổi Fourier ngược
    F_inverse_shift = np.fft.ifftshift(F_filtered_shift)
    img_back = np.fft.ifft2(F_inverse_shift)
    img_filtered = np.abs(img_back)
    
    # Ép kiểu dải giá trị về chuẩn ảnh xám uint8 (0-255)
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    
    # 7. Tính toán phổ biên độ log chuẩn hóa để hiển thị đồ thị / GUI
    spectrum_orig = np.log(1 + np.abs(F_shift))
    spectrum_orig = cv2.normalize(spectrum_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    spectrum_filt = np.log(1 + np.abs(F_filtered_shift))
    spectrum_filt = cv2.normalize(spectrum_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_filtered, spectrum_orig, spectrum_filt, mask








