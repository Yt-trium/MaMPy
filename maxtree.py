import matplotlib.pyplot as plt
import numpy as np
import imageio

# read an image as a 2D array of 8 bit integer (grayscale)
def image_read(filename):
    return imageio.imread(uri=filename, as_gray=True).astype(dtype=np.uint8)


if __name__ == '__main__':
    img1 = image_read(filename="examples/images/cameraman.jpg")

    print(img1)
    print(img1.shape)
    print(type(img1))
    print(type(img1[0][0]))
    print(img1.max())
    print(img1.min())

    plt.imshow(img1, cmap="gray")
    plt.show()


    img2 = image_read(filename="examples/images/lapin.jpg")

    print(img2)
    print(img2.shape)
    print(type(img2))
    print(type(img2[0][0]))
    print(img2.max())
    print(img2.min())

    plt.imshow(img2, cmap="gray")
    plt.show()