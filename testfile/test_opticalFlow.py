import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.opticalFlowEstimation import horn_schunck_optical_flow, visualize_flow_quiver

# 1. Tạo dữ liệu giả lập chuyển động (Nếu không có video/ảnh thực tế)
# Khung hình 1: Khối hộp ở vị trí (150, 150)
img1 = np.ones((400, 400), dtype=np.uint8) * 50
cv2.rectangle(img1, (150, 150), (250, 250), 200, -1)
# Thêm chút cấu hình hạt để tính đạo hàm mượt hơn
np.random.seed(42)
img1 = np.clip(img1 + np.random.normal(0, 2, img1.shape), 0, 255).astype(np.uint8)

# Khung hình 2: Khối hộp dịch chuyển chéo xuống dưới sang phải (+10 pixel)
img2 = np.ones((400, 400), dtype=np.uint8) * 50
cv2.rectangle(img2, (160, 160), (260, 260), 200, -1)
img2 = np.clip(img2 + np.random.normal(0, 2, img2.shape), 0, 255).astype(np.uint8)

# 2. Thực thi thuật toán Horn-Schunck với alpha=1.0 và 150 bước lặp
u, v = horn_schunck_optical_flow(img1, img2, alpha=1.0, num_iters=150)

# 3. Trực quan hóa vector chuyển động dưới dạng sơ đồ mũi tên
result_img = visualize_flow_quiver(img2, u, v, step=12)

# 4. Hiển thị đồ thị kết quả báo cáo
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Frame 1 (Khung trước)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title("Frame 2 (Khung sau)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title("Trường vận tốc Horn-Schunck (Quiver)")
plt.axis('off')

plt.tight_layout()
plt.show()