import cv2
import numpy as np
import matplotlib.pyplot as plt


def rescale_greylevels(src, g_values=16):
    """
    Foliensatz: MuML_BV_01d_Bilder_und_Bildeigenschaften
    Folie: 20
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


def smooth_mean(src, kernel_size=3, fast=False):
    """
    Foliensatz: MuML_BV_01d_Bilder_und_Bildeigenschaften
    Folie: 22
    Applies mean filtering (smoothing) to a greyscale image using a square kernel.

    The function supports two modes:
    - Manual implementation (slow but educational), using nested loops and wrapping borders
    - Fast mode using OpenCV's built-in cv2.blur function

    The filter computes the mean value of a square neighborhood of size `kernel_size × kernel_size`
    for each pixel and replaces the center pixel with that mean. Wrapping is used at the image borders
    to simulate circular boundary behavior.

    :param src: Input greyscale image as a NumPy array (dtype: uint8)
    :param kernel_size: Size of the smoothing kernel (must be odd), default is 3
    :param fast: If True, uses OpenCV's cv2.blur for faster execution;
                 if False, uses manual implementation (default: False)
    :return: Smoothed image as a NumPy array (same shape and dtype as input)
    """
    img = src.copy()

    pad = kernel_size // 2

    padded = cv2.copyMakeBorder(img, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_WRAP)

    if not fast:
        rows, columns = img.shape[:2]

        for row in range(rows):
            for column in range(columns):
                mean = 0.0
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        mean += padded[row + i, column + j]
                mean = mean / (kernel_size ** 2)
                img[row, column] = int(round(mean))
    else:
        img = cv2.blur(img, (kernel_size, kernel_size))

    return img


def set_contrast(src, gain):
    """
    Foliensatz: MuML_BV_01d_Bilder_und_Bildeigenschaften
    Folie: 25
    Adjusts the contrast of a greyscale image using a linear contrast transformation.

    The contrast is modified by scaling each pixel value relative to the image's mean grey level:
    new_value = (value - mean) * gain + mean

    - If gain > 1: contrast is increased (dark gets darker, bright gets brighter)
    - If gain < 1: contrast is reduced (image becomes flatter)
    - If gain = 1: no change

    After the transformation, pixel values are clipped to the valid range [0, 255].

    A histogram of the resulting image is displayed using matplotlib.

    :param src: Input greyscale image as a NumPy array (dtype: uint8)
    :param gain: Contrast scaling factor (float > 0)
    :return: Contrast-adjusted image as a NumPy array (dtype: uint8)
    """
    img = src.astype(np.float32)

    rows, columns = img.shape
    mean = 0.0
    for row in range(rows):
        for column in range(columns):
            mean += img[row, column]
    mean = mean / (rows * columns)

    for row in range(rows):
        for column in range(columns):
            img[row, column] = (img[row, column] - mean) * gain + mean

    img = np.clip(img, 0, 255)

    plt.hist(img.ravel(), bins=256, range=(0, 256))
    plt.title("Histogramm")
    plt.xlabel("Grauwert")
    plt.ylabel("Anzahl Pixel")
    plt.show()

    return img.astype(np.uint8)


