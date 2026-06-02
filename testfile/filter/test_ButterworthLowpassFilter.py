import numpy as np
import cv2

img = cv2.imread("data/images/powerOfTwo.jpg",0)

F = np.fft.fft2(img)
F = np.fft.fftshift(F)

rows, cols = img.shape
crow = rows // 2
ccol = cols // 2
mask = np.zeros((rows, cols))

Y, X = np.ogrid[:rows, :cols]

distance = np.sqrt(
    (Y-crow)**2 +
    (X-ccol)**2
)


D0 = 50
n = 10
# n 1-10

mask = 1 / (
    1 + (distance / D0)**(2*n)
)


F_butter = F * mask

img_butter = np.fft.ifft2(
    np.fft.ifftshift(F_butter)
)

img_butter = np.abs(img_butter)


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(122)
plt.imshow(img_butter,cmap='gray')
plt.title("Lowpass")

plt.show()