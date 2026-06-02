import cv2
import matplotlib.pyplot as plt
from src.filters import ideal_lowpass_filter

# 1. Đọc ảnh mẫu
img = cv2.imread("data/images/powerOfTwo.jpg", 0)
if img is None:
    # Dự phòng nếu không tìm thấy file ảnh
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.putText(img, "HUST", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)

# 2. Gọi hàm xử lý với bán kính mong muốn (ví dụ: D0 = 40)
img_out, spec_orig, spec_filt, filter_mask = ideal_lowpass_filter(img, radius=40)

# 3. Hiển thị kết quả lưới 2x2 đúng chuẩn cấu trúc báo cáo của Student A
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("1. Ảnh gốc (Original)")
axes[0, 0].axis('off')

axes[0, 1].imshow(spec_orig, cmap='gray')
axes[0, 1].set_title("2. Phổ tần số gốc (Original Spectrum)")
axes[0, 1].axis('off')

axes[1, 0].imshow(filter_mask, cmap='gray')
axes[1, 0].set_title("3. Mặt nạ Ideal Lowpass Filter")
axes[1, 0].axis('off')

axes[1, 1].imshow(img_out, cmap='gray')
axes[1, 1].set_title("4. Ảnh sau khi lọc (Lowpass Image)")
axes[1, 1].axis('off')

plt.tight_layout()
# Lưu kết quả đồ thị phục vụ Filter Gallery trong báo cáo
plt.savefig("results_ideal_lowpass.png", bbox_inches='tight', dpi=300)
plt.show()