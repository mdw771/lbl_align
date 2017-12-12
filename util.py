import numpy as np
from scipy.ndimage import fourier_shift
from pyfftw.interfaces.numpy_fft import fftn, ifftn


def realign_image(arr, shift, angle=0):
    """
    Translate and rotate image via Fourier

    Parameters
    ----------
    arr : ndarray
        Image array.

    shift: float
        Mininum and maximum values to rescale data.

    angle: float, optional
        Mininum and maximum values to rescale data.

    Returns
    -------
    ndarray
        Output array.
    """
    # if both shifts are integers, do circular shift; otherwise perform Fourier shift.
    if np.count_nonzero(np.abs(np.array(shift) - np.round(shift)) < 0.01) == 2:
        temp = np.roll(arr, int(shift[0]), axis=0)
        temp = np.roll(temp, int(shift[1]), axis=1)
        temp = temp.astype('float32')
    else:
        temp = fourier_shift(np.fft.fftn(arr), shift)
        temp = np.fft.ifftn(temp)
        temp = np.abs(temp).astype('float32')
    return temp


def realign_image_1d(arr, shift, angle=0):

    # if both shifts are integers, do circular shift; otherwise perform Fourier shift.
    if np.abs(np.array(shift) - np.round(shift)) < 0.01:
        temp = np.roll(arr, int(shift))
        temp = temp.astype('float32')
    else:
        temp = fourier_shift(fftn(arr), shift)
        temp = ifftn(temp)
        temp = np.abs(temp).astype('float32')
    return temp