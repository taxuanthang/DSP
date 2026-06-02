import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh và chuyển sang float để tránh tràn số uint8
img = cv2.imread("data/images/normal2.png", 0)
if img is None:
    raise FileNotFoundError("Không tìm thấy ảnh. Vui lòng kiểm tra lại đường dẫn!")

img = img.astype(np.float64)
rows, cols = img.shape

# 2. Tạo nhiễu sọc tuần hoàn theo trục X
x = np.arange(cols)
frequency_period = 20  # Chu kỳ nhiễu (20 pixel)
noise = 50 * np.sin(2 * np.pi * x / frequency_period)
noise = np.tile(noise, (rows, 1))
img_noisy = img + noise

# 3. Biến đổi Fourier 2D sang miền tần số
F = np.fft.fft2(img_noisy)
F_shift = np.fft.fftshift(F)

# 4. Tính toán và hiển thị phổ biên độ (để trực quan hóa)
mag = np.abs(F_shift)
mag_normalized = mag / np.max(mag)
spectrum = np.log(1 + 1000 * mag_normalized)
spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
gamma = 0.3 
spectrum = np.power(spectrum, gamma)

# 5. Khởi tạo Mặt nạ lọc Notch (Mặc định giữ lại toàn bộ bằng 1)
mask = np.ones((rows, cols))

def add_notch(mask, center_y, center_x, radius):
    Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
    distance = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
    mask[distance <= radius] = 0
    return mask

# 6. TỰ ĐỘNG TÍNH TOÀN VỊ TRÍ NOTCH THEO TÂM ẢNH
# Vì nhiễu sọc tạo theo hàm sin nằm ngang, các điểm nhiễu sẽ lệch đối xứng qua tâm theo trục X
center_y = rows // 2
center_x = cols // 2

# Khoảng cách từ tâm đến đỉnh nhiễu phụ thuộc vào chu kỳ tần số của hàm sin
# Công thức dịch dịch tần số: offset = cols / frequency_period
offset_x = int(cols / frequency_period)

# Đặt 2 điểm notch đối xứng qua tâm (bán kính bằng 10 như bạn thiết lập)
mask = add_notch(mask, center_y, center_x - offset_x, radius=10)
mask = add_notch(mask, center_y, center_x + offset_x, radius=10)

# 7. Áp dụng bộ lọc Notch trong miền tần số
F_notch = F_shift * mask

# 8. Biến đổi Fourier ngược để phục hồi lại ảnh sạch
img_back_shift = np.fft.ifftshift(F_notch)
img_notch = np.fft.ifft2(img_back_shift)
img_notch = np.abs(img_notch)

# Giới hạn dải giá trị về [0, 255] để hiển thị chuẩn xác
img_notch = np.clip(img_notch, 0, 255).astype(np.uint8)
img_noisy_display = np.clip(img_noisy, 0, 255).astype(np.uint8)

# 9. Hiển thị kết quả bằng Subplots theo chuẩn báo cáo môn học
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(img_noisy_display, cmap='gray')
axes[0, 0].set_title("Ảnh bị nhiễu sọc (Noisy Image)")

axes[0, 1].imshow(spectrum, cmap='gray')
axes[0, 1].set_title("Phổ tần số gốc (Original Spectrum)")

# Vẽ hiển thị mặt nạ lọc để dễ kiểm tra xem đã đặt trúng điểm trắng của nhiễu chưa
axes[1, 0].imshow(mask, cmap='gray')
axes[1, 0].set_title("Mặt nạ lọc Notch (Notch Filter Mask)")

axes[1, 1].imshow(img_notch, cmap='gray')
axes[1, 1].set_title("Ảnh sau khi lọc sạch sọc (Filtered Image)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()