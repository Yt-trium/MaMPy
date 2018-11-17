import scipy as scp
import scipy.misc

import numpy as np


def image_read(filename):
    img = (scp.misc.imread(name=filename, flatten=True, mode="L")).astype(dtype=np.uint8)


if __name__ == '__main__':
    image_read(filename="examples/images/cameraman.jpg")
    