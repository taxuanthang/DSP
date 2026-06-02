import cv2
import numpy as np

img = cv2.imread("data/images/normal2.png",0)

F = np.fft.fft2(img)
F = np.fft.fftshift(F)

rows, cols = img.shape
crow = rows // 2
ccol = cols // 2
mask = np.zeros((rows, cols))
radius = 50

Y, X = np.ogrid[:rows, :cols]

distance = np.sqrt(
    (Y-crow)**2 +
    (X-ccol)**2
)

D0 = 10

mask = 1 - np.exp(
    -(distance**2)/(2*(D0**2))
)


F_gaussian = F * mask

img_gaussian = np.fft.ifft2(
    np.fft.ifftshift(F_gaussian)
)

img_gaussian = np.abs(img_gaussian)


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(122)
plt.imshow(img_gaussian,cmap='gray')
plt.title("Highpass")

plt.show()