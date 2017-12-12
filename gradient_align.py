import dxchange
import numpy as np
from util import *



def align_gradient(img, search_range=(-10, 10, 0.5)):
    """
    Align images line-by-line using gradient minimization.
    :param img: np.ndarray
           Input image.
    :param search_range: tuple
           (search start, search end, search step).
    :return: np.ndarray
             Aligned image.
    """

    accum = 0
    dat = np.copy(img)
    offset_ls = np.arange(*search_range)
    for i in range(dat.shape[0]):
        if i > 0:
            grad_ls = []
            this_line = dat[i]
            last_line = dat[i-1]
            if accum != 0:
                this_line = realign_image_1d(this_line, accum)
            for dx in offset_ls:
                temp = realign_image_1d(this_line, dx)
                edge_l = 0
                edge_r = len(this_line)
                if dx < 0:
                    edge_r += int(np.floor(dx))
                elif dx > 0:
                    edge_l += int(np.ceil(dx))
                grad = np.sum(np.abs(temp[edge_l:edge_r] - last_line[edge_l:edge_r]))
                grad /= edge_r - edge_l
                grad_ls.append(grad)
            shift = offset_ls[np.argmin(grad_ls)]
            accum += shift
            dat[i] = realign_image_1d(this_line, shift)
    return dat


if __name__ == '__main__':

    img = dxchange.read_tiff('data/scan154_HDPC.tif')
    res = align_gradient(img, search_range=(-10, 10, 0.5))
    dxchange.write_tiff(res, 'data/out', dtype='float32', overwrite=True)