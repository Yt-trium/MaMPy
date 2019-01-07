"""
Maxtree without Union-by-rank

Reference:
    A fair comparison of many max-tree computation algorithms
    (Extended version of the paper submitted to ISMM 2013)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import imageio
import sys

'''
  * read an image as a 2D array of 8 bit integer (grayscale)
'''
def image_read(filename):
    return imageio.imread(uri=filename, as_gray=True).astype(dtype=np.uint8)

def find_pixel_parent(parents, index):
    """
    Given an image containing pixel's parent and a pixel id,
    returns the id of its parent id.

    The parent is also named as root.
    A pixel is the root of itself if parents[index] == index.
    """

    root = parents[index]

    # Assign the root of the given pixel to the root of its parent.
    if root != index:
        parents[index] = find_pixel_parent(parents, root)
        return parents[index]
    else:
        return root


def canonize(image, parents, nodes_order):
    """
    Makes sure all nodes of a max tree are valid.
    """

    for pi in nodes_order:
        root = parents[pi]

        if image[root] == image[parents[root]]:
            parents[pi] = parents[root]


def get_4_neighbors(width, height, resolution, pi, pixel_row):
    """
    For a given image width, height and pixel index, return the index
    of direct neighbor pixels using 4 connection.
    """

    top_pi = pi - width 
    bottom_pi = pi + width
    left_pi = pi - 1
    right_pi = pi + 1

    neighbors = []

    if top_pi >= 0:
        neighbors.append(top_pi)
    if bottom_pi < resolution:
        neighbors.append(bottom_pi) 

    # For right and left pixels, we need to check if we moved to 
    # the next row or not.
    if math.floor(left_pi / width) == pixel_row:
        neighbors.append(left_pi) 
    if math.floor(right_pi / width) == pixel_row:
        neighbors.append(right_pi)

    return neighbors


def get_8_neighbors(width, height, resolution, pi, pixel_row):
    """
    For a given image width, height and pixel index, return the index
    of direct neighbor pixels using 8 connection.
    """

    neighbors = get_4_neighbors(width, height, resolution, pi, pixel_row)

    top_left_pi = pi - width  - 1
    top_right_pi = top_left_pi + 2
    bottom_left_pi = pi + width - 1
    bottom_right_pi = bottom_left_pi + 2

    if top_left_pi >= 0 and math.floor(top_left_pi / width) == pixel_row - 1:
        neighbors.append(top_left_pi)

    if top_right_pi >= 0 and math.floor(top_right_pi / width) == pixel_row - 1:
        neighbors.append(top_right_pi)

    if bottom_left_pi < resolution and math.floor(bottom_left_pi / width) == pixel_row + 1:
        neighbors.append(bottom_left_pi)

    if bottom_right_pi < resolution and math.floor(bottom_right_pi / width) == pixel_row + 1:
        neighbors.append(bottom_right_pi)

    return neighbors


def maxtree_berger(image, connection8=True):
    """
    Union-find based max-tree algorithm as proposed by Berger et al.

    -> Algorithm 2 in the paper.

    Arguments:
    image is supposed to be a numpy array.

    Returns:
    """

    (width, height) = (image.shape[0], image.shape[1])

    flatten_image = image.flatten()
    resolution = flatten_image.shape[0]

    # Unique value telling if a pixel is defined in the max tree or not.
    undefined_node = resolution + 2

    # We generate an extra vector of pixels that order nodes downard.
    # This vector allow to traverse the tree both upward and downard
    # without having to sort childrens of each node.
    # Initially, we sort pixel by increasing value and add indices in it.
    sorted_pixels = flatten_image.argsort()

    # We store in the parent node of each pixel in an image.
    # To do so we use the index of the pixel (x + y * width).
    parents = np.full(
        resolution, 
        fill_value=undefined_node, 
        dtype=np.uint32)

    # zparents make root finding much faster.
    zparents = parents.copy()

    # We go through sorted pixels in the reverse order.
    for pi in reversed(sorted_pixels):
        # Make a node.
        # By default, a pixel is its own parent.
        parents[pi] = pi
        zparents[pi] = pi

        # Find the row of this pixel.
        pixel_row = math.floor(pi / width)

        # We need to go through neighbors that already have a parent.
        if connection8:
            neighbors = get_8_neighbors(width, height, resolution, pi, pixel_row)
        else:
            neighbors = get_4_neighbors(width, height, resolution, pi, pixel_row)

        # Filter neighbors.
        neighbors = [n for n in neighbors if parents[n] != undefined_node]

        # Merge nodes together.
        for nei_pi in neighbors:
            nei_root = find_pixel_parent(zparents, nei_pi)

            if nei_root != pi:
                zparents[nei_root] = pi
                parents[nei_root] = pi

    canonize(flatten_image, parents, sorted_pixels)
    return parents, sorted_pixels


def maxtree_berger_rank(image, connection8=True):
    """
    Union-find with union-by-rank based max-tree algorithm .

    -> Algorithm 3 in the paper.

    Arguments:
    image is supposed to be a numpy array.

    Returns:
    """

    (width, height) = (image.shape[0], image.shape[1])

    flatten_image = image.flatten()
    resolution = flatten_image.shape[0]

    # Unique value telling if a pixel is defined in the max tree or not.
    undefined_node = resolution + 2

    # We generate an extra vector of pixels that order nodes downard.
    # This vector allow to traverse the tree both upward and downard
    # without having to sort childrens of each node.
    # Initially, we sort pixel by increasing value and add indices in it.
    sorted_pixels = flatten_image.argsort()

    # We store in the parent node of each pixel in an image.
    # To do so we use the index of the pixel (x + y * width).
    parents = np.full(
        resolution,
        fill_value=undefined_node,
        dtype=np.uint32)

    ranks = np.full(
        resolution,
        fill_value=0,
        dtype=np.uint32)

    reprs = np.full(
        resolution,
        fill_value=0,
        dtype=np.uint32)

    # zparents make root finding much faster.
    zparents = parents.copy()

    # We go through sorted pixels in the reverse order.
    for pi in reversed(sorted_pixels):
        # Make a node.
        # By default, a pixel is its own parent.
        parents[pi] = pi
        zparents[pi] = pi
        ranks[pi] = 0
        reprs[pi] = 0

        zp = pi

        # Find the row of this pixel.
        pixel_row = math.floor(pi / width)

        # We need to go through neighbors that already have a parent.
        if connection8:
            neighbors = get_8_neighbors(width, height, resolution, pi, pixel_row)
        else:
            neighbors = get_4_neighbors(width, height, resolution, pi, pixel_row)

        # Filter neighbors.
        neighbors = [n for n in neighbors if parents[n] != undefined_node]

        # Go through neighbors.
        for nei_pi in neighbors:
            zn = find_pixel_parent(zparents, nei_pi)

            if zn != zp:
                parents[reprs[zn]] = pi

                if ranks[zp] < ranks[zn]:
                    # Swap them.
                    zp, zn = zn, zp

                # Merge sets.
                zparents[zn] = zp
                reprs[zp] = pi

                if ranks[zp] == ranks[zn]:
                    ranks[zp] += 1

    canonize(flatten_image, parents, sorted_pixels)

    return parents, sorted_pixels


def maxtree_union_find_level_compression(image, connection8=True):
    """
    Union-find with level compression.

    -> Algorithm 5 in the paper.

    Arguments:
    image is supposed to be a numpy array.

    Returns:
    """

    (width, height) = (image.shape[0], image.shape[1])

    flatten_image = image.flatten()
    resolution = flatten_image.shape[0]

    # Unique value telling if a pixel is defined in the max tree or not.
    undefined_node = resolution + 2

    # We generate an extra vector of pixels that order nodes downard.
    # This vector allow to traverse the tree both upward and downard
    # without having to sort childrens of each node.
    # Initially, we sort pixel by increasing value and add indices in it.
    sorted_pixels = flatten_image.argsort()

    # We store in the parent node of each pixel in an image.
    # To do so we use the index of the pixel (x + y * width).
    parents = np.full(
        resolution,
        fill_value=undefined_node,
        dtype=np.uint32)

    # zparents make root finding much faster.
    zparents = parents.copy()

    j = resolution - 1

    # We go through sorted pixels in the reverse order.

    for i in reversed(range(len(sorted_pixels))):
        pi = sorted_pixels[i]
        # Make a node.
        # By default, a pixel is its own parent.
        parents[pi] = pi
        zparents[pi] = pi

        zp = pi

        # Find the row of this pixel.
        pixel_row = math.floor(pi / width)

        # We need to go through neighbors that already have a parent.
        if connection8:
            neighbors = get_8_neighbors(width, height, resolution, pi, pixel_row)
        else:
            neighbors = get_4_neighbors(width, height, resolution, pi, pixel_row)

        # Filter neighbors.
        neighbors = [n for n in neighbors if parents[n] != undefined_node]

        # Go through neighbors.
        for nei_pi in neighbors:
            zn = find_pixel_parent(zparents, nei_pi)

            if zn != zp:
                if flatten_image[zp] == flatten_image[zn]:
                    zp, zn = zn, zp

                # Merge sets.
                zparents[zn] = zp
                parents[zn] = zp

                sorted_pixels[j] = zn
                j -= 1

    canonize(flatten_image, parents, sorted_pixels)

    return parents, sorted_pixels

def compute_attribute_area(s, parent, ima):
    # Image should be flattened.
    resolution = ima.shape[0]

    attr = np.full(
        resolution,
        fill_value=1,
        dtype=np.uint32)

    proot = s[0]

    for pi in reversed(s):
        if (pi == proot):
            continue
        q = parent[pi]
        attr[q] += attr[pi]

    return attr

def direct_filter(s, parent, ima, attr, λ):
    # Image should be flattened.
    resolution = ima.shape[0]

    out = np.full(
        resolution,
        fill_value=0,
        dtype=np.uint32)

    proot = s[0]

    if attr[proot] < λ:
        out[proot] = 0
    else:
        out[proot] = ima[proot]

    for pi in s:
        q = parent[pi]

        if ima[q] == ima[pi]:
            out[pi] = out[q]
        elif attr[pi] < λ:
            out[pi] = out[q]
        else:
            out[pi] = ima[q]

    return out

if __name__ == '__main__':
    file_path = sys.argv[1]
    area_filter = int(sys.argv[2])
    img1 = image_read(filename=file_path)

    print(img1.shape)

    flatten_image = img1.flatten()
    (parents, s) = maxtree_berger(img1, connection8=False)
    attr = compute_attribute_area(s, parents, flatten_image)
    edited = direct_filter(s, parents, flatten_image, attr, area_filter)
    edited = np.reshape(edited, img1.shape)

    fig, (fig1, fig2) = plt.subplots(1, 2)
    fig1.set_axis_off();
    fig2.set_axis_off();
    fig1.set_title("Original")
    fig2.set_title("After")
    fig1.imshow(img1, cmap="gray")
    fig2.imshow(edited, cmap="gray")
    plt.show()
