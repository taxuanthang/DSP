import cv2
import matplotlib.pyplot as plt
from src.filters import gaussian_highpass_filter

# 1. Đọc ảnh mẫu
img = cv2.imread("data/images/normal2.png", 0)
if img is None:
    import numpy as np
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(img, (150, 150), (350, 350), 255, -1)

# 2. Gọi hàm xử lý với bán kính D0 = 10 đúng theo code của bạn
img_out, spec_orig, spec_filt, filter_mask = gaussian_highpass_filter(img, radius=10)

# 3. Hiển thị kết quả lưới 2x2 để quan sát đầy đủ cấu trúc
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("1. Ảnh gốc (Original)")

axes[0, 1].imshow(spec_orig, cmap='gray')
axes[0, 1].set_title("2. Phổ tần số gốc")

axes[1, 0].imshow(filter_mask, cmap='gray')
axes[1, 0].set_title("3. Mặt nạ Gaussian Highpass (D0=10)")

axes[1, 1].imshow(img_out, cmap='gray')
axes[1, 1].set_title("4. Ảnh sau lọc (Highpass)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.savefig("results_gaussian_highpass.png", bbox_inches='tight', dpi=300)
plt.show()