"""
Common utilities functions.
C. Meyer
"""

import imageio
import numpy as np

def image_read(filename):
    """
    Read an image as a 2D array of 8 bit integers (grayscale pixels)
    """
    return imageio.imread(uri=filename, as_gray=True).astype(dtype=np.uint8)
