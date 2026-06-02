from src import fft2
import numpy as np

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load ảnh
img = cv2.imread("data/images/test2.jpg", 0)

plt.imshow(img, cmap='gray')
plt.show()


# Pad ảnh
img = fft2.pad_to_power_of_two(img)


# FFT2 ảnh
custom_fft = fft2.fft2_custom(img)

numpy_fft = np.fft.fft2(img)

print(np.allclose(custom_fft, numpy_fft))


# Visualize
fft_shifted = np.fft.fftshift(custom_fft)
magnitude = np.abs(fft_shifted)
spectrum = np.log(1 + magnitude)
plt.imshow(spectrum, cmap='gray')
plt.title("Frequency Spectrum")
plt.show()


#Tính sai số 
print(
    np.max(
        np.abs(custom_fft - numpy_fft)
    )
)