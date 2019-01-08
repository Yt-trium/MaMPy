import matplotlib.pyplot as plt
import numpy as np
import imageio

from algo2 import maxtree_berger, maxtree_berger_rank, maxtree_union_find_level_compression

'''
  * read an image as a 2D array of 8 bit integer (grayscale)
'''
def image_read(filename):
    return imageio.imread(uri=filename, as_gray=True).astype(dtype=np.uint8)

'''
  * return the max tree of an given 2D image
'''
def maxtree(image):

    return 0

if __name__ == '__main__':
    '''
    image test 1
    
    [110,  90, 100]
    [ 50,  50,  50]
    [ 40,  20,  50]
    [ 50,  50,  50]
    [120,  70,  80] 
    
            20
            |
            40
            |
            50
         /      \
       90        70
      /  \      /  \
    110  100  120  80
    
    '''
    '''
    img_test_1 = np.array([[110, 90, 100], [50, 50, 50], [40, 20, 50], [50, 50, 50], [120, 70, 80]], dtype=np.uint8)


    print(img_test_1)
    print(img_test_1.shape)
    print(type(img_test_1))
    print(type(img_test_1[0][0]))
    print(img_test_1.max())
    print(img_test_1.min())

    (parents, sorted_pixels) = maxtree_berger_rank(img_test_1)
    parents = parents.reshape(img_test_1.shape)
    print(parents)
    print(sorted_pixels)

    plt.imshow(img_test_1, cmap="gray")
    plt.show()
    '''

    '''
    image test 2

    [15, 13, 16]
    [12, 12, 10]
    [16, 12, 14]
    '''

    img_test_2 = np.array([[15, 13, 16], [12, 12, 10], [16, 12, 14]],dtype=np.uint8)

    print(img_test_2)
    print(img_test_2.shape)
    print(type(img_test_2))
    print(type(img_test_2[0][0]))
    print(img_test_2.max())
    print(img_test_2.min())

    (parents, sorted_pixels) = maxtree_berger(img_test_2, False)
    parents = parents.reshape(img_test_2.shape)
    print(parents)
    print(sorted_pixels)

    plt.imshow(img_test_2, cmap="gray")
    plt.show()

    '''
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
    '''
