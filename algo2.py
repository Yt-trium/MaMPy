"""
Maxtree without Union-by-rank

Reference:
    A fair comparison of many max-tree computation algorithms
    (Extended version of the paper submitted to ISMM 2013)
"""

import numpy as np
import scipy as scp
import scipy.misc

def find_pixel_parent(parents, index):
    """
    Given an image containing pixel's parent and a pixel id,
    returns the id of its parent id.

    The parent is also named as root.
    """

    root = parents[index]

    # Assign the root of the given pixel to the root of its parent.
    if (root != index):
        parents[index] = find_pixel_parent(parents, root)

    return parents[index]

def maxtree_berger(image):
    """
    Union-find based max-tree algorithm as proposed by Berger et al.

    Arguments:
    image is supposed to be a numpy array.

    Returns:
    """

    flatten_image = image.flatten()
    resolution = flatten_image.shape[0]

    # We store in the parent node of each pixel in an image.
    # To do so we use the index of the pixel (x + y * width).
    parent = np.full(shape=(resolution, 1), fill_value=-1, dtype=np.int32)

    # We generate an extra vector of pixels that order nodes downard.
    # This vector allow to traverse the tree both upward and downard
    # without having to sort childrens of each node.
    # Initially, we sort pixel by increasing value and add indices in it.
    nodes = np.arange(resolution, dtype=np.uint32)
    # TODO: sort using pixel values

    print(nodes)


    return (parent)

if __name__ == '__main__':
    maxtree_berger(np.zeros((200, 400), dtype=np.uint8))
