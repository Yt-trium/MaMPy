"""
Tree of Shape computation and filters.

Reference:
    [1] Un algorithme de complexité linéaire pour le calcul de l’arbre des formes.
    Edwin Carlinet, Sébastien Crozet, Thierry Géraud
    [2] A quasi-linear algorithm to compute the tree of shapes of n-D images
    Thierry Géraud, Edwin Carlinet, Sébastien Crozet, Laurent Najman.

Authors:
C. Meyer
"""

import numpy as np
from enum import Enum

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
    if InterpolationMode == InterpolationMode.MAX:
        interpolationFunc = np.min
    elif InterpolationMode == InterpolationMode.MIN:
        interpolationFunc = np.max
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
    return 0

# interpolation + immersion
# 1. add median outside
# 2. interpolate
# 3. immerse

print(interpolate2D(np.array([[1, 2], [3, 4]]), InterpolationMode.MAX))

