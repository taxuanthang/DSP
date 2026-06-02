import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.filters import ideal_highpass_filter

# 1. Đọc ảnh mẫu
img = cv2.imread("data/images/normal2.png", 0)
if img is None:
    # Tạo ảnh giả lập nếu không tìm thấy file ảnh thực tế
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(img, (128, 128), (384, 384), 255, -1) # Vẽ hình vuông trắng giữa nền đen để sinh cạnh sắc

# 2. Gọi hàm xử lý với bán kính cắt radius = 10
img_out, spec_orig, spec_filt, filter_mask = ideal_highpass_filter(img, radius=10)

# 3. Hiển thị kết quả lưới 2x2 đúng cấu trúc báo cáo của Student A
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("1. Ảnh gốc (Original)")
axes[0, 1].imshow(spec_orig, cmap='gray')
axes[0, 1].set_title("2. Phổ tần số gốc")

axes[1, 0].imshow(filter_mask, cmap='gray')
axes[1, 0].set_title("3. Mặt nạ Ideal Highpass Mask")
axes[1, 1].imshow(img_out, cmap='gray')
axes[1, 1].set_title("4. Ảnh lọc thông cao (Cạnh sắc)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
# Lưu ảnh kết quả trực tiếp ra file phục vụ báo cáo
plt.savefig("results_ideal_highpass.png", bbox_inches='tight', dpi=300)
plt.show()