"""
Tree of Shape computation and filters.

Reference:
    [1] Un algorithme de complexité linéaire pour le calcul de l’arbre des formes.
    Edwin Carlinet, Sébastien Crozet, Thierry Géraud
    [2] A quasi-linear algorithm to compute the tree of shapes of n-D images
    Thierry Géraud, Edwin Carlinet, Sébastien Crozet, Laurent Najman.

Authors:
C. Meyer

Preconditions:
Images are 2d single 8-bit channel arrays.

Todo:
Work with 3d images too.
"""

from collections import deque
import numpy as np
from enum import Enum

def addBorderMedian(input):
    '''
    :param input: numpy 2d array of a single channel image
    :return: numpy 2d array of the same image with a border equal to the median of the original border.
    '''

    # check input validity
    assert type(input) == np.ndarray
    assert input.ndim == 2

    # create output
    output = np.zeros(tuple([x + 2 for x in input.shape]), dtype=input.dtype)

    border = np.zeros(2*input.shape[0] + 2*input.shape[1] - 4, dtype=input.dtype)
    b = 0

    for x in range(input.shape[0]):
        for y in range(input.shape[1]):
            if(x == 0 or x == input.shape[0]-1 or y == 0 or y == input.shape[1]-1):
                border[b] = input[x][y]
                b += 1

    med = np.median(border)

    for x in range(output.shape[0]):
        for y in range(output.shape[1]):
            if(x == 0 or x == output.shape[0]-1 or y == 0 or y == output.shape[1]-1):
                output[x][y] = med
            else:
                output[x][y] = input[x-1][y-1]

    return output


InterpolationMode = Enum('InterpolationMode', 'MAX MIN MED')

def interpolate2D(input, interpolationMode):
    """
    :param input: numpy 2d array of a single channel image
    :param interpolationMode: the interpolation mode (max, min or median)
    :return: numpy 2d array of the interpolated image
    """

    # check input validity
    assert type(input) == np.ndarray
    assert type(interpolationMode) == InterpolationMode
    assert input.ndim == 2

    # create output
    output = np.zeros(tuple([2 * x - 1 for x in input.shape]), dtype=input.dtype)

    # select interpolation function
    if interpolationMode == InterpolationMode.MAX:
        interpolationFunc = np.max
    elif interpolationMode == InterpolationMode.MIN:
        interpolationFunc = np.min
    else:
        interpolationFunc = np.median

    for x in range(input.shape[0]):
        for y in range(input.shape[1]):
            # copy input
            output[2 * x][2 * y] = input[x][y]

            if(x < input.shape[0]-1):
                # interpolation x + 1
                output[2 * x + 1][2 * y] = interpolationFunc([input[x][y], input[x + 1][y]])
            if(y < input.shape[1]-1):
                # interpolation y + 1
                output[2 * x][2 * y + 1] = interpolationFunc([input[x][y], input[x][y + 1]])

            if(x < input.shape[0]-1 and y < input.shape[1]-1):
                # interpolation x + 1 and y + 1
                output[2 * x + 1][2 * y + 1] =interpolationFunc([input[x][y], input[x + 1][y],
                                                                input[x][y + 1], input[x + 1][y + 1]])

    return output

def immersion2D(input):
    '''
    :param input: numpy 2d array of a single channel image
    :return: numpy 2d array of the immersed image
    '''

    # check input validity
    assert type(input) == np.ndarray
    assert input.ndim == 2

    # create output
    output = np.zeros(tuple([2 * x - 1 for x in input.shape]), dtype=object)

    for x in range(input.shape[0]):
        for y in range(input.shape[1]):
            # copy input
            output[2 * x][2 * y] = input[x][y]

            if(x < input.shape[0]-1):
                # interpolation x + 1
                output[2 * x + 1][2 * y] = (np.min([input[x][y], input[x + 1][y]]),
                                            np.max([input[x][y], input[x + 1][y]]))
            if(y < input.shape[1]-1):
                # interpolation y + 1
                output[2 * x][2 * y + 1] = (np.min([input[x][y], input[x][y + 1]]),
                                            np.max([input[x][y], input[x][y + 1]]))

            if(x < input.shape[0]-1 and y < input.shape[1]-1):
                # interpolation x + 1 and y + 1
                output[2 * x + 1][2 * y + 1] = (np.min([input[x][y], input[x + 1][y],
                                                        input[x][y + 1], input[x + 1][y + 1]]),
                                                np.max([input[x][y], input[x + 1][y],
                                                        input[x][y + 1], input[x + 1][y + 1]]),)

    return output


def interpolateAndImmerse2D(input, interpolationMode):
    '''
    :param input: numpy 2d array of a single channel image
    :param interpolationMode: the interpolation mode (max, min or median)
    :return: numpy 2d array of the Khalimsky grid
    '''

    return immersion2D(interpolate2D(addBorderMedian(input), interpolationMode))


def priorityPush(q, h, U, l):
    if type(U[h]) == tuple:
        lower = U[h][0]
        upper = U[h][1]
    else:
        lower = U[h]
        upper = U[h]

    if lower > l:
        l_ = lower
    elif upper < l:
        l_ = upper
    else:
        l_ = l

    q[l_].append(h)


def priorityPop(q, l):
    # empty queue
    local_l = l
    l_ = -1
    if len(q[l]) == 0:
        for i in range(1, max(l, 255-l)):
            if len(q[l-i]) > 0:
                l_ = l-i
                break
            elif len(q[l+i]) > 0:
                l_ = l+i
                break

        local_l = l_

    return q[local_l].popleft()


def q_empty(hierarchical_queue):
    for i in range(0, 255):
        if len(hierarchical_queue[i]) > 0:
            return False
    return True


import max_tree


def sort(input):
    '''
    :param input: numpy 2d array of a single 8-bit channel image
    :return:
    '''
    input_flat = input.flatten()
    resolution = input_flat.size

    u = np.ndarray(resolution, dtype=input.dtype)
    r = np.ndarray(resolution, dtype=input.dtype)

    deja_vu = np.ndarray(resolution, dtype=bool)
    deja_vu.fill(False)

    # Create queue
    hierarchical_queue = np.ndarray(256, dtype=object)
    for i in range(0, 256):
        hierarchical_queue[i] = deque()

    i = 0

    hierarchical_queue[input_flat[0]].append(0)
    deja_vu[0] = True

    level = input_flat[0]

    while not q_empty(hierarchical_queue):
        h = priorityPop(hierarchical_queue, level)
        u[h] = level
        r[i] = h

        neighbors = max_tree.get_neighbors_2d(4, input.shape, h, input.size)
        neighbors = [n for n in neighbors if not deja_vu[n]]

        for n in neighbors:
            priorityPush(hierarchical_queue, n, input_flat, level)
            deja_vu[n] = True

        i = i + 1

    return r.reshape(input.shape), u.reshape(input.shape)


def main():
    test = immersion2D(interpolate2D(np.array([[1, 2], [3, 4]]), InterpolationMode.MAX))

    print(test)

    print(sort(test))

    # print(interpolateAndImmerse2D(np.array([[1, 2], [3, 4]]), InterpolationMode.MAX))


if __name__ == "__main__":
    main()
