import cv2
import numpy as np


def downscale(src, g_values=16):
    img = src.copy()
