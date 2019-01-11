"""
Common utilities functions.
C. Meyer
"""

import imageio
import numpy as np

"""
 * Read an image as a 2D array of 8 bit integers (grayscale pixels)
"""
def image_read(filename):
    return imageio.imread(uri=filename, as_gray=True).astype(dtype=np.uint8)
