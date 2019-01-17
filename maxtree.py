"""
Max-Tree computation and filters.

Reference:
    [1] A fair comparison of many max-tree computation algorithms
    (Extended version of the paper submitted to ISMM 2013)
    Edwin Carlinet, Thierry Géraud.

Authors:
K. Masson
C. Meyer
"""

import numpy as np
import numba
from numba import jit
import math
import logging
from collections import namedtuple

# MaMPy includes
# Utilities
from utils import image_read


# Named Tuple for the max-tree structure
MaxTreeStructure = namedtuple("MaxTreeStructure", ["parent", "S"])

@jit(nopython=True)
def maxtree_berger_rank(image, connection8=True):
    """
    Union-find with union-by-rank based max-tree algorithm .

    -> Algorithm 3 in the paper.

    Arguments:
    image is supposed to be a numpy array.

    Returns:
    """

    (height, width) = (image.shape[0], image.shape[1])

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
    for pi in sorted_pixels[::-1]:
        # Make a node.
        # By default, a pixel is its own parent.
        parents[pi] = pi
        zparents[pi] = pi
        ranks[pi] = 0
        reprs[pi] = pi

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

    return MaxTreeStructure(parents, sorted_pixels)




@jit(nopython=True)
def find_pixel_parent(parents, index):
    """
    Given an image containing pixel's parent and a pixel id, returns the id of its parent id.
    The parent is also named as root. A pixel is the root of itself if parents[index] == index.
    """

    root = parents[index]

    # Assign the root of the given pixel to the root of its parent.
    if root != index:
        parents[index] = find_pixel_parent(parents, root)
        return parents[index]
    else:
        return root


@jit(nopython=True)
def canonize(image, parents, nodes_order):
    """
    Makes sure all nodes of a max tree are valid.
    """

    for pi in nodes_order:
        root = parents[pi]

        if image[root] == image[parents[root]]:
            parents[pi] = parents[root]


@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
def maxtree_berger(image, connection8=True):
    """
    Union-find (without union-by-rank) based max-tree algorithm as proposed by Berger et al.

    -> Algorithm 2 in the paper.

    Arguments:
    image is supposed to be a numpy array.

    Returns:
    """

    (height, width) = (image.shape[0], image.shape[1])

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
    for pi in sorted_pixels[::-1]:
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



@jit(nopython=True)
def maxtree_union_find_level_compression(image, connection8=True):
    """
    Union-find with level compression.

    -> Algorithm 5 in the paper.

    Arguments:
    image is supposed to be a numpy array.

    Returns:
    """

    (height, width) = (image.shape[0], image.shape[1])

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
    for pi in sorted_pixels[::-1]:
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



@jit(nopython=True)
def direct_filter(maxtree_p_s, input, attribute, λ):
    """
    The parameters order follows the order given in the article [1]
    :param maxtree_p_s:
    :param input:
    :param attribute:
    :param λ:
    :return:
    """

    ima = input.flatten()

    out = np.full(
        ima.shape,
        fill_value=0,
        dtype=input.dtype)

    proot = maxtree_p_s.S[0]

    if attribute[proot] < λ:
        out[proot] = 0
    else:
        out[proot] = ima[proot]

    for p in maxtree_p_s.S:
        q = maxtree_p_s.parent[p]

        if ima[q] == ima[p]:
            out[p] = out[q]     # p not canonical
        elif attribute[p] < λ:
            out[p] = out[q]     # Criterion failed
        else:
            out[p] = ima[p]     # Criterion pass

    return out.reshape(input.shape)



@jit(nopython=True)
def maxtree(input, connexity=None):
    """
    Compute the max-tree of a 2D/3D image using simple strategy to select algorithms
    (Check the "A comparison of many max-tree computation algorithms" poster for the decision tree strategy)
    This is the only max-tree computing function the regular user should use

    :param input: numpy ndarray of a single channel image
                  Good practice : use a numpy fixed-size dtype (https://www.numpy.org/devdocs/user/basics.types.html)
    :param connexity: connexity of the maxtree : acceptable value : 2D 4/8 - 3D 6/18/26 (default 4 and 6)
    :return: the maxtree of the image (parent and S vector pair)
    """
    # Check input
    if input.ndim not in [2, 3]:
        raise ValueError("Input image is not a 2D or 3D array")

    if input.ndim == 2:
        if connexity is None or connexity not in [4, 8]:
            connexity = 4
            # print("Connexity set to default (2D -> 4-connexity)")
    if input.ndim == 3:
        if connexity is None or connexity not in [6, 18, 26]:
            connexity = 6
            # print("Connexity set to default (3D -> 6-connexity)")

    # Choose algorithm
    # Low quantization (<= 8-bit pixels) : Salembier et al.
    # /!\ Not Implemented Yet /!\
    # if input.dtype in [np.bool, np.byte, np.ubyte, np.int8, np.uint8]:
    # maxtree_salembier(input, connexity)

    # High quantization : Berger + rank
    return maxtree_berger_rank(input, connexity)


@jit(nopython=True)
def area_filter(input, threshold, maxtree_p_s=None):
    """
    :param input: numpy ndarray of a single channel image
    :param threshold: threshold of the filter (minimum area to keep)
    :param maxtree_p_s: the maxtree of the image (parent and S vector pair)
    :return: numpy ndarray of the image
    """
    # Check input
    if input.ndim not in [2, 3]:
        raise ValueError("Input image is not a 2D or 3D array")

    if threshold < 1:
        raise ValueError("Threshold less than 1")

    if maxtree_p_s is None:
        maxtree_p_s = maxtree(input)

    if maxtree_p_s.parent.size != maxtree_p_s.S.size:
        raise ValueError("Invalid max-tree")

    if maxtree_p_s.S.size != input.size:
        raise ValueError("Image and max-tree doesn't match")

    # Compute area attribute
    area_attribute = np.full(input.size,
                             fill_value=1,
                             dtype=np.uint32)

    # Everything except the first item, reversed
    # > np.arange(8)[:0:-1]
    # array([7, 6, 5, 4, 3, 2, 1])
    for p in maxtree_p_s.S[:0:-1]:
        q = maxtree_p_s.parent[p]
        area_attribute[q] += area_attribute[p]

    # Apply Filter
    return direct_filter(maxtree_p_s, input, area_attribute, threshold)


@jit(nopython=True)
def contrast_filter(input, threshold, maxtree_p_s=None):
    """
    :param input: numpy ndarray of a single channel image
    :param threshold: threshold of the filter (minimum contrast to keep)
    :param maxtree_p_s: the maxtree of the image (parent and S vector pair)
    :return: numpy ndarray of the image
    """
    # Check input
    if input.ndim not in [2, 3]:
        raise ValueError("Input image is not a 2D or 3D array")

    if threshold < 1:
        raise ValueError("Threshold less than 1")

    if maxtree_p_s is None:
        maxtree_p_s = maxtree(input)

    if maxtree_p_s.parent.size != maxtree_p_s.S.size:
        raise ValueError("Invalid max-tree")

    if maxtree_p_s.S.size != input.size:
        raise ValueError("Image and max-tree doesn't match")

    # Compute contrast attribute
    pixel_values = input.flatten()
    contrast_attribute = np.full(input.size,
                             fill_value=0,
                             dtype=np.float64)

    # Everything except the first item, reversed
    # > np.arange(8)[:0:-1]
    # array([7, 6, 5, 4, 3, 2, 1])
    for p in maxtree_p_s.S[:0:-1]:
        q = maxtree_p_s.parent[p]
        contrast_attribute[q] = max(contrast_attribute[q], pixel_values[p] - pixel_values[q])

    # Apply Filter
    return direct_filter(maxtree_p_s, input, contrast_attribute, threshold)

"""
image_input = image_read("examples/images/circuit_small.png")
# image_input = image_input / image_input.max()
# image_output = area_filter(image_input, 500)
image_output = contrast_filter(image_input, 1)


import matplotlib.pyplot as plt
plt.imshow(image_input, cmap="gray")
plt.show()
plt.imshow(image_output, cmap="gray")
plt.show()

print(image_input.max(), image_input.min(), image_input.max() - image_input.min())

"""