import numpy as np
from src import fft1d

x = np.array([1,2,3,4])

print(fft1d.fft_recursive(x))
print(np.fft.fft(x))

np.allclose(fft1d.fft_recursive(x),np.fft.fft(x))
