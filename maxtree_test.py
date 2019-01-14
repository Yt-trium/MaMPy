import maxtree as mtlib
import numpy as np
import math
import unittest

class TestFindPixelParentFunction(unittest.TestCase):
    """Test behavior of find_pixel_parent funciton."""

    def setUp(self):
        self.parents = np.array([3, 1, 0, 1, 0, 2])
        """
        A tree reprensatation of this image is the following:

                               1
                               |
                               3
                               |
                               0
                              / \
                             2   4
                             |
                             5
        """

    def test_out_of_bound_pixel_index(self):
        """We expect it to fail."""

        parents = self.parents.copy()

        with self.assertRaises(IndexError):
            mtlib.find_pixel_parent(parents, 42)

    def test_negative_pixel_index(self):
        """We expect it to fail."""

        parents = self.parents.copy()

        # Warning: in numpy you can fetch in both ways.
        self.assertEqual(
            mtlib.find_pixel_parent(parents, -3),
            mtlib.find_pixel_parent(parents, len(parents) - 3))
        self.assertEqual(
            mtlib.find_pixel_parent(parents, -1),
            mtlib.find_pixel_parent(parents, len(parents) -1))

    def test_wrong_parents_data(self):
        """The parents image should contains pixel valid indices"""

        wrong_parents = np.array([0, 3, 2, 10, 49, 50])

        # We expect it to fail.
        with self.assertRaises(IndexError):
            mtlib.find_pixel_parent(wrong_parents, 3)

    def test_parents(self):
        parents = self.parents.copy()
        self.assertEqual(mtlib.find_pixel_parent(parents, 1), 1)
        self.assertEqual(mtlib.find_pixel_parent(parents, 3), 1)
        self.assertEqual(mtlib.find_pixel_parent(parents, 0), 1)
        self.assertEqual(mtlib.find_pixel_parent(parents, 2), 1)
        self.assertEqual(mtlib.find_pixel_parent(parents, 4), 1)
        self.assertEqual(mtlib.find_pixel_parent(parents, 5), 1)


class TestCanonizeFunction(unittest.TestCase):
    """Test behavior of canonize funciton."""

    def setUp(self):
        # Pixel values.
        # Values based on the paper:
        #   A fair comparison of many max-tree computation algorithms.
        self.image = np.array([15, 13, 16, 12, 12, 10, 16, 12, 14])
        self.parents = np.array([1, 4, 1, 4, 5, 5, 4, 4, 4])
        self.nodes = np.array([5, 4, 1, 7, 8, 3, 2, 6, 0])

    def test_canonize_final_result(self):
        """
        Canonizing a correct result shouldn't do anything.
        """
        parents = self.parents.copy()

        mtlib.canonize(self.image, parents, self.nodes)

        self.assertEqual(parents[0], 1)
        self.assertEqual(parents[1], 4)
        self.assertEqual(parents[2], 1)
        self.assertEqual(parents[3], 4)
        self.assertEqual(parents[4], 5)
        self.assertEqual(parents[5], 5)
        self.assertEqual(parents[6], 4)
        self.assertEqual(parents[7], 4)
        self.assertEqual(parents[8], 4)

    def test_canonize_unmerged_parents(self):
        """
        Pixels having the same value should be merged together.
        """

        parents = np.array([1, 4, 1, 7, 5, 5, 4, 4, 4])

        mtlib.canonize(self.image, parents, self.nodes)

        self.assertEqual(parents[0], 1)
        self.assertEqual(parents[1], 4)
        self.assertEqual(parents[2], 1)
        self.assertEqual(parents[3], 4)
        self.assertEqual(parents[4], 5)
        self.assertEqual(parents[5], 5)
        self.assertEqual(parents[6], 4)
        self.assertEqual(parents[7], 4)
        self.assertEqual(parents[8], 4)

class TestNeighborFetching(unittest.TestCase):
    """Test results when getting neighbor of a pixel"""

    def test_fetch_empty_image(self):
        with self.assertRaises(ZeroDivisionError):
            mtlib.get_4_neighbors(0, 0, 0, 0, 0)

        with self.assertRaises(ZeroDivisionError):
            mtlib.get_8_neighbors(0, 0, 0, 0, 0)

    def test_row_calculation(self):
        """
        Formula: pixel_row = math.floor(pixel_index / image_width)
        """

        self.assertEqual(math.floor(0 / 100), 0)
        self.assertEqual(math.floor(50 / 100), 0)
        self.assertEqual(math.floor(99 / 100), 0)
        self.assertEqual(math.floor(100 / 100), 1)
        self.assertEqual(math.floor(0 / 1), 0)
        self.assertEqual(math.floor(1 / 1), 1)
        self.assertEqual(math.floor(2 / 1), 2)
        self.assertEqual(math.floor(0 / 20), 0)
        self.assertEqual(math.floor(39 / 20), 1)
        self.assertEqual(math.floor(40 / 20), 2)

    def test_one_row_image(self):
        width = 100
        height = 1
        resolution = 100
        pi = 20
        row = math.floor(pi / width)

        n4 = mtlib.get_4_neighbors(width, height, resolution, pi, row)
        n8 = mtlib.get_8_neighbors(width, height, resolution, pi, row)

        self.assertEqual(2, len(n4))
        self.assertEqual(19, n4[0])
        self.assertEqual(21, n4[1])
        self.assertEqual(n4, n8)

    def test_one_col_image(self):
        width = 1
        height = 100
        resolution = 100
        pi = 40
        row = math.floor(pi / width)

        n4 = mtlib.get_4_neighbors(width, height, resolution, pi, row)
        n8 = mtlib.get_8_neighbors(width, height, resolution, pi, row)

        self.assertEqual(2, len(n4))
        self.assertEqual(39, n4[0])
        self.assertEqual(41, n4[1])
        self.assertEqual(n4, n8)

    def test_upper_left_corner(self):
        width = 20
        height = 10
        resolution = 200
        pi = 0
        row = 0

        n4 = mtlib.get_4_neighbors(width, height, resolution, pi, row)
        n8 = mtlib.get_8_neighbors(width, height, resolution, pi, row)

        self.assertEqual(2, len(n4))
        self.assertTrue(1 in n4)
        self.assertTrue(20 in n4)

        self.assertEqual(3, len(n8))
        self.assertTrue(all(pi in n8 for pi in n4))
        self.assertTrue(21 in n8)

    def test_upper_right_corner(self):
        width = 20
        height = 10
        resolution = 200
        pi = 19
        row = 0

        n4 = mtlib.get_4_neighbors(width, height, resolution, pi, row)
        n8 = mtlib.get_8_neighbors(width, height, resolution, pi, row)

        self.assertEqual(2, len(n4))
        self.assertTrue(18 in n4)
        self.assertTrue(39 in n4)

        self.assertEqual(3, len(n8))
        self.assertTrue(all(pi in n8 for pi in n4))
        self.assertTrue(38 in n8)

    def test_lower_left_corner(self):
        width = 20
        height = 10
        resolution = 200
        pi = 180
        row = 9

        n4 = mtlib.get_4_neighbors(width, height, resolution, pi, row)
        n8 = mtlib.get_8_neighbors(width, height, resolution, pi, row)

        self.assertEqual(2, len(n4))
        self.assertTrue(181 in n4)
        self.assertTrue(160 in n4)

        self.assertEqual(3, len(n8))
        self.assertTrue(all(pi in n8 for pi in n4))
        self.assertTrue(161 in n8)

    def test_lower_left_corner(self):
        width = 20
        height = 10
        resolution = 200
        pi = 199
        row = 9

        n4 = mtlib.get_4_neighbors(width, height, resolution, pi, row)
        n8 = mtlib.get_8_neighbors(width, height, resolution, pi, row)

        self.assertEqual(2, len(n4))
        self.assertTrue(198 in n4)
        self.assertTrue(179 in n4)

        self.assertEqual(3, len(n8))
        self.assertTrue(all(pi in n8 for pi in n4))
        self.assertTrue(178 in n8)

    def test_fetch_inside(self):
        width = 20
        height = 10
        resolution = 200
        pi = 50
        row = math.floor(pi / width)

        n4 = mtlib.get_4_neighbors(width, height, resolution, pi, row)
        n8 = mtlib.get_8_neighbors(width, height, resolution, pi, row)

        self.assertEqual(4, len(n4))
        self.assertTrue(49 in n4)
        self.assertTrue(51 in n4)
        self.assertTrue(30 in n4)
        self.assertTrue(70 in n4)

        self.assertEqual(8, len(n8))
        self.assertTrue(71 in n8)
        self.assertTrue(69 in n8)
        self.assertTrue(29 in n8)
        self.assertTrue(31 in n8)
        self.assertTrue(all(pi in n8 for pi in n4))


"""
class TestBergerMaxTree(unittest.TestCase):
    Test results of berger maxtree funciton.

    def setUp(self):
        self.image = np.array(
            [[15, 13, 16], 
            [12, 12, 10], 
            [16, 12, 14]])

    def test_maxtree(self):
        (parents, nodes) = mtlib.maxtree_berger(self.image, connection8=True)

        print(parents)
        print(nodes)

        (parents, nodes) = mtlib.maxtree_berger_rank(self.image, connection8=False)

        print(parents)
        print(nodes)

        (parents, nodes) = mtlib.maxtree_union_find_level_compression(self.image, connection8=False)

        print(parents)
        print(nodes)
"""


if __name__ == '__main__':
    unittest.main()

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

    img_test_2 = np.array([[15, 13, 16], [12, 12, 10], [16, 12, 14]], dtype=np.uint8)

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

