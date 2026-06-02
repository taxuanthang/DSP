import cv2
import os
import matplotlib.pyplot as plt
from src.degradation import apply_motion_blur

# 1. Đọc ảnh mẫu sạch ban đầu
img = cv2.imread("data/images/normal2.png", 0)
if img is None:
    import numpy as np
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(img, "HUST DSP", (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4)

# 2. Gọi hàm đóng gói tạo nhòe (L=30, độ lệch chuẩn nhiễu=5.0)
L = 30
img_deg_float, img_deg_uint8, F_psf = apply_motion_blur(img, length=L, noise_sigma=5.0)

# 3. Tạo thư mục chứa kết quả nếu chưa có và lưu ảnh bằng OpenCV
os.makedirs("results", exist_ok=True)
cv2.imwrite(f"results/motion_blur_L{L}.png", img_deg_uint8)
print(f"Đã lưu ảnh nhòe 'results/motion_blur_L{L}.png' thành công!")

# 4. Hiển thị so sánh trực quan
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original (Clean Image)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_deg_uint8, cmap='gray')
plt.title(f"Degraded Image (Motion Blur L={L} + Noise)")
plt.axis('off')

plt.tight_layout()
plt.show()