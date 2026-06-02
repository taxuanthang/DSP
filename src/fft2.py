from src import fft1d 
import numpy as np

def fft2_rows(img):

    rows, cols = img.shape

    result = np.zeros((rows, cols), dtype=complex)

    for r in range(rows):
        result[r,:] = fft1d.fft_recursive(img[r,:])

    return result

def fft2_custom(img):
    
    rows, cols = img.shape

    if rows & (rows - 1):
        raise ValueError(f"FFT requires power-of-two length, got {rows}")

    if cols & (cols - 1):
        raise ValueError(f"FFT requires power-of-two length, got {cols}")

    row_fft = fft2_rows(img)

    rows, cols = row_fft.shape

    result = np.zeros((rows, cols), dtype=complex)

    for c in range(cols):
        result[:,c] = fft1d.fft_recursive(row_fft[:,c])

    return result

def pad_to_power_of_two(img):
    h, w = img.shape

    new_h = 1 << (h - 1).bit_length()
    new_w = 1 << (w - 1).bit_length()

    padded = np.zeros((new_h, new_w), dtype=img.dtype)
    padded[:h, :w] = img

    return padded