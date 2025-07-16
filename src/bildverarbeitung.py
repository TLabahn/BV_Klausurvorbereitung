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


def get_hist(src):
    """
    Foliensatz: MuML_BV_01d_Bilder_und_Bildeigenschaften
    Folie: 26
    Extracts all greyscale pixel values from the input image into a flat list.

    This function iterates over each pixel in the image and returns a list of their values.
    It is typically used to generate histogram data (e.g., with matplotlib).

    Note: The image is assumed to be a single-channel greyscale image.

    :param src: Input greyscale image as a NumPy array (dtype: uint8)
    :return: List of pixel values (integers from 0 to 255)
    """
    img = src.copy()

    rows, columns = img.shape[:2]

    histogram = []

    for row in range(rows):
        for column in range(columns):
            histogram.append(img[row, column])

    return histogram


def aufgabe_1():
    img = np.array([[0, 2, 1, 3, 0],
                    [0, 2, 3, 2, 2],
                    [0, 0, 0, 3, 0],
                    [0, 0, 1, 3, 0],
                    [0, 1, 2, 1, 0]])
    img = img.astype(np.uint8)

    hist = cv2.calcHist(
        images=[img],
        channels=[0],
        mask=None,
        histSize=[4],
        ranges=[0, 4]
    )
    hist = hist.flatten()

    t = 1
    N = np.sum(hist[:])
    N_H = np.sum(hist[:t])
    N_V = np.sum(hist[t:])

    W_H = N_H / N
    W_V = N_V / N

    graustufen = np.arange(len(hist))
    mu_H = np.sum(graustufen[:t] * hist[:t]) / N_H
    mu_V = np.sum(graustufen[t:] * hist[t:]) / N_V

    sigma2H = np.sum((graustufen[:t] - mu_H) ** 2 * hist[:t]) / N_H
    sigma2V = np.sum((graustufen[t:] - mu_V) ** 2 * hist[t:]) / N_V

    sigma2W = W_H * sigma2H + W_V * sigma2V

    print(f'N: {N}\n'
          f'N_H: {N_H}\n'
          f'N_V: {N_V}\n'
          f'W_H: {W_H}\n'
          f'W_V: {W_V}\n'
          f'mu_H: {mu_H}\n'
          f'mu_V: {mu_V}\n'
          f'sigma2H: {sigma2H}\n'
          f'sigma2V: {sigma2V}\n'
          f'sigma2W: {sigma2W}\n'
          f'Histogram: {hist}\n'
          f'Graustufen: {graustufen}')








