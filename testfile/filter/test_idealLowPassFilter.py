import cv2
import numpy as np

img = cv2.imread("data/images/powerOfTwo.jpg",0)

F = np.fft.fft2(img)
F = np.fft.fftshift(F)

rows, cols = img.shape
crow = rows // 2
ccol = cols // 2
mask = np.zeros((rows, cols))
radius = 50

for i in range(rows):
    for j in range(cols):

        distance = np.sqrt(
            (i-crow)**2 +
            (j-ccol)**2
        )

        if distance <= radius:
            mask[i,j] = 1


F_filtered = F * mask

F_inverse = np.fft.ifftshift(F_filtered)
img_filtered = np.fft.ifft2(F_inverse)
img_filtered = np.abs(img_filtered)


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(122)
plt.imshow(img_filtered,cmap='gray')
plt.title("Lowpass")

plt.show()


# Hiển thị Spectrum

spectrum = np.log(
    1 + np.abs(F)
)

spectrum_filtered = np.log(
    1 + np.abs(F_filtered)
)