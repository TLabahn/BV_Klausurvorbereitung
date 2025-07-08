import cv2
import numpy as np


def rescale_greylevels(src, g_values=16):
    """
    Rescales the grey levels of a greyscale image to a specified number of discrete grey values.

    This function maps the pixel values of a greyscale image to `g_values` uniformly distributed levels
    over the full possible range defined by the image's bit depth.

    This works both for downscaling (e.g. reducing an 8-bit image to 16 levels) and upscaling
    (e.g. expanding 16-level data to fill the 8-bit range while preserving the number of grey levels).

    The resulting image has pixel values between 0 and `g_values - 1`, stretched to fit the full output type range.

    :param src: Input greyscale image as a NumPy array (dtype: uint8 or uint16)
    :param g_values: Number of target grey levels (default: 16)
    :return: Rescaled image as a NumPy array (dtype: uint8), with pixel values in [0, g_values - 1]
    """
    img = src.copy()

    bits = img.itemsize * 8
    img = (img / (2 ** bits - 1) * (g_values - 1)).astype(np.uint8)

    return img

