import numpy as np

def fft_recursive(x):

    N = len(x)

    if N == 1:
        return x

    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])

    factor = np.exp(-2j*np.pi*np.arange(N)/N)

    return np.concatenate(
        [
            even + factor[:N//2]*odd,
            even + factor[N//2:]*odd
        ]
    )