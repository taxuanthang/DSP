import cv2
import matplotlib.pyplot as plt
from src.filters import butterworth_lowpass_filter

# 1. Đọc ảnh mẫu
img = cv2.imread("data/images/powerOfTwo.jpg", 0)
if img is None:
    import numpy as np
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(img, "HUST DSP", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4)

# 2. Gọi hàm xử lý với bán kính D0 = 50 và bậc n = 10 đúng như bạn chọn
img_out, spec_orig, spec_filt, filter_mask = butterworth_lowpass_filter(img, radius=50, n=10)

# 3. Hiển thị kết quả lưới 2x2 chuẩn báo cáo
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("1. Ảnh gốc (Original)")

axes[0, 1].imshow(spec_orig, cmap='gray')
axes[0, 1].set_title("2. Phổ tần số gốc")

axes[1, 0].imshow(filter_mask, cmap='gray')
axes[1, 0].set_title("3. Mặt nạ Butterworth Lowpass (n=10)")

axes[1, 1].imshow(img_out, cmap='gray')
axes[1, 1].set_title("4. Ảnh sau khi lọc (Lowpass)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.savefig("results_butterworth_lowpass.png", bbox_inches='tight', dpi=300)
plt.show()