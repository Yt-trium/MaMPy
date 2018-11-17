import numpy as np

import scipy as scp
import scipy.misc

def image_read(filename):
    return (scp.misc.imread(name=filename, flatten=True, mode="L")).astype(dtype=np.uint8)

if __name__ == '__main__':
    img1 = image_read(filename="examples/images/cameraman.jpg")

    print(img1)
    print(img1.shape)
    print(type(img1))
    print(type(img1[0][0]))
    print(img1.max())
    print(img1.min())
