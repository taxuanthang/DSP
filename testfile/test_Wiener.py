import cv2
import os
import matplotlib.pyplot as plt
from src.degradation import apply_motion_blur, wiener_deconvolution

# 1. Đọc ảnh gốc sạch ban đầu
img_origin = cv2.imread("data/images/normal2.png", 0)
if img_origin is None:
    import numpy as np
    img_origin = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(img_origin, "HUST DSP", (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4)

# 2. MỤC 9: Tạo ảnh bị nhòe chuyển động và cộng nhiễu (L=30)
print("--- BẮT ĐẦU QUÁ TRÌNH TẠO NHÒE MOTION BLUR ---")
img_deg_float, img_deg_uint8, F_psf = apply_motion_blur(img_origin, length=30, noise_sigma=0.2)

# 3. MỤC 10: Thực hiện lọc khôi phục Wiener và quét tìm K tối ưu
print("\n--- BẮT ĐẦU QUÁ TRÌNH LỌC WIENER & QUÉT TỐI ƯU SSIM ---")
K_list = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
best_img, opt_K, max_ssim = wiener_deconvolution(img_deg_float, F_psf, img_origin, K_candidates=K_list)

print(f"\n=> KẾT QUẢ TỐI ƯU NHẤT: K = {opt_K} với SSIM = {max_ssim:.4f}")

# 4. Lưu ảnh khôi phục tốt nhất bằng OpenCV
os.makedirs("results", exist_ok=True)
cv2.imwrite(f"results/wiener_restored_K{opt_K}.png", best_img)
print(f"Đã lưu ảnh khôi phục thành công tại 'results/wiener_restored_K{opt_K}.png'")

# 5. Hiển thị lưới đồ thị 1 hàng 3 cột so sánh tiến trình khôi phục
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_origin, cmap='gray')
plt.title("1. Ảnh gốc ban đầu (Clean)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_deg_uint8, cmap='gray')
plt.title("2. Ảnh lỗi (Blur + Noise)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(best_img, cmap='gray')
plt.title(f"3. Khôi phục Wiener (Tối ưu K={opt_K})")
plt.axis('off')

plt.tight_layout()
plt.show()