#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Tests of the MaMPy library
C. Meyer
"""

import matplotlib.pyplot as plt
from utils import image_read
import max_tree
import tree_of_shape

"""
I. Max-Tree
"""

#image_input = image_read("examples/images/circuit_small_small.png")
#image_output = max_tree.area_filter(image_input, 100)

#plt.imshow(image_input, cmap="gray")
#plt.show()
#plt.imshow(image_output, cmap="gray")
#plt.show()

"""
II. Tree of shapes
"""

image_input = image_read("examples/images/circuit_small_small.png")
plt.imshow(image_input, cmap="gray")
plt.show()

image_output = tree_of_shape.area_filter(image_input, 5)
plt.imshow(image_output, cmap="gray")
plt.show()

image_output = tree_of_shape.area_filter(image_input, 20)
plt.imshow(image_output, cmap="gray")
plt.show()


"""
III. Binary partition tree
"""
