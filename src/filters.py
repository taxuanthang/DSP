import cv2
import numpy as np
from src.fft2 import  fft2_custom, ifft2_custom , pad_to_power_of_two



def ideal_lowpass_filter(img, radius=50):
    src = img.astype(np.float64)
    # Pad ảnh lên lũy thừa của 2 để tránh lỗi kích thước
    padded_src, orig_h, orig_w = pad_to_power_of_two(src)
    h, w = padded_src.shape
    
    # Sử dụng fft2_custom và dịch tâm phổ
    f_shift = np.fft.fftshift(fft2_custom(padded_src))
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    mask = np.zeros((h, w))
    mask[d <= radius] = 1.0
    
    f_out = f_shift * mask
    
    # Sử dụng ifft2_custom đưa về miền không gian ảnh
    img_back = ifft2_custom(np.fft.ifftshift(f_out))
    
    # Cắt (crop) ảnh về kích thước gốc ban đầu
    img_back_cropped = img_back[:orig_h, :orig_w]
    res = np.clip(np.abs(img_back_cropped), 0, 255).astype(np.uint8)
    
    # Map phổ để hiển thị GUI (giữ nguyên kích thước sau khi pad hoặc crop tùy bạn, ở đây crop để đồng bộ)
    f_shift_crop = f_shift[:orig_h, :orig_w]
    f_out_crop = f_out[:orig_h, :orig_w]
    mask_crop = mask[:orig_h, :orig_w]
    
    spec_orig = np.log(1 + np.abs(f_shift_crop))
    spec_orig = cv2.normalize(spec_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    spec_filt = np.log(1 + np.abs(f_out_crop))
    spec_filt = cv2.normalize(spec_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask_crop

# --- BỘ LỌC THÔNG CAO LÝ TƯỞNG (IDEAL HIGHPASS) ---
def ideal_highpass_filter(img, radius=10):
    src = img.astype(np.float64)
    padded_src, orig_h, orig_w = pad_to_power_of_two(src)
    h, w = padded_src.shape
    
    f_shift = np.fft.fftshift(fft2_custom(padded_src))
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    mask = np.ones((h, w))
    mask[d <= radius] = 0.0
    
    f_out = f_shift * mask
    
    img_back = ifft2_custom(np.fft.ifftshift(f_out))
    img_back_cropped = img_back[:orig_h, :orig_w]
    res = cv2.normalize(np.abs(img_back_cropped), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    f_shift_crop = f_shift[:orig_h, :orig_w]
    f_out_crop = f_out[:orig_h, :orig_w]
    mask_crop = mask[:orig_h, :orig_w]
    
    spec_orig = cv2.normalize(np.log(1 + np.abs(f_shift_crop)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_filt = cv2.normalize(np.log(1 + np.abs(f_out_crop)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask_crop

# --- BỘ LỌC THÔNG THẤP BUTTERWORTH ---
def butterworth_lowpass_filter(img, radius=50, n=2):
    src = img.astype(np.float64)
    padded_src, orig_h, orig_w = pad_to_power_of_two(src)
    h, w = padded_src.shape
    
    f_shift = np.fft.fftshift(fft2_custom(padded_src))
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    mask = 1.0 / (1.0 + (d / radius) ** (2 * n))
    
    f_out = f_shift * mask
    
    img_back = ifft2_custom(np.fft.ifftshift(f_out))
    img_back_cropped = img_back[:orig_h, :orig_w]
    res = np.clip(np.abs(img_back_cropped), 0, 255).astype(np.uint8)
    
    f_shift_crop = f_shift[:orig_h, :orig_w]
    f_out_crop = f_out[:orig_h, :orig_w]
    mask_crop = mask[:orig_h, :orig_w]
    
    spec_orig = cv2.normalize(np.log(1 + np.abs(f_shift_crop)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_filt = cv2.normalize(np.log(1 + np.abs(f_out_crop)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask_crop

# --- BỘ LỌC THÔNG CAO GAUSSIAN ---
def gaussian_highpass_filter(img, radius=10):
    src = img.astype(np.float64)
    padded_src, orig_h, orig_w = pad_to_power_of_two(src)
    h, w = padded_src.shape
    
    f_shift = np.fft.fftshift(fft2_custom(padded_src))
    
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    mask = 1.0 - np.exp(-(d ** 2) / (2 * (radius ** 2)))
    
    f_out = f_shift * mask
    
    img_back = ifft2_custom(np.fft.ifftshift(f_out))
    img_back_cropped = img_back[:orig_h, :orig_w]
    res = np.clip(np.abs(img_back_cropped), 0, 255).astype(np.uint8)
    
    f_shift_crop = f_shift[:orig_h, :orig_w]
    f_out_crop = f_out[:orig_h, :orig_w]
    mask_crop = mask[:orig_h, :orig_w]
    
    spec_orig = cv2.normalize(np.log(1 + np.abs(f_shift_crop)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spec_filt = cv2.normalize(np.log(1 + np.abs(f_out_crop)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return res, spec_orig, spec_filt, mask_crop

# --- BỘ LỌC CHẶN DẢI NOTCH (NOTCH REJECT FILTER) ---
def notch_reject_filter(img, uk, vk, radius=10):
    src = img.astype(np.float64)
    padded_src, orig_h, orig_w = pad_to_power_of_two(src)
    h, w = padded_src.shape
    
    f_shift = np.fft.fftshift(fft2_custom(padded_src))
    mask = np.ones((h, w), dtype=np.float64)
    cy, cx = h // 2, w // 2
    
    def clear_noise_spot(m_matrix, py, px, r):
        Y, X = np.ogrid[:m_matrix.shape[0], :m_matrix.shape[1]]
        dist = np.sqrt((Y - py)**2 + (X - px)**2)
        m_matrix[dist <= r] = 0
        return m_matrix

    if uk != 30 or vk != 30:
        mask = clear_noise_spot(mask, cy + vk, cx + uk, r=radius)
        mask = clear_noise_spot(mask, cy - vk, cx - uk, r=radius)
    else:
        period = 20
        offset = int(w / period)
        mask = clear_noise_spot(mask, cy, cx - offset, r=radius)
        mask = clear_noise_spot(mask, cy, cx + offset, r=radius)
        
    f_notch = f_shift * mask
    
    img_back = ifft2_custom(np.fft.ifftshift(f_notch))
    img_back_cropped = img_back[:orig_h, :orig_w]
    res = np.clip(np.abs(img_back_cropped), 0, 255).astype(np.uint8)
    
    # Chỉ cắt phần phổ đã xử lý để hiển thị tương thích kích thước cũ
    f_notch_crop = f_notch[:orig_h, :orig_w]
    
    return res, None, f_notch_crop, None