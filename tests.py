import algo2 as mtlib
import numpy as np
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

class TestBergerMaxTree(unittest.TestCase):
    """Test results of berger maxtree funciton."""

    def setUp(self):
        self.image = np.array(
            [[15, 13, 16], 
            [12, 12, 10], 
            [16, 12, 14]])

    def test_maxtree(self):
        """
        Values based on the paper:
           A fair comparison of many max-tree computation algorithms.
        """

        (parents, nodes) = mtlib.maxtree_berger(self.image, connection8=False)

        fparents = parents.flatten()
        fnodes = nodes.flatten()

        self.assertEqual(fnodes[0], 5)
        self.assertEqual(fnodes[1], 4)
        self.assertEqual(fnodes[2], 1)
        self.assertEqual(fnodes[3], 7)
        self.assertEqual(fnodes[4], 8)
        self.assertEqual(fnodes[5], 3)
        self.assertEqual(fnodes[6], 2)
        self.assertEqual(fnodes[7], 6)
        self.assertEqual(fnodes[8], 0)

        self.assertEqual(fparents[0], 1)
        self.assertEqual(fparents[1], 4)
        self.assertEqual(fparents[2], 1)
        self.assertEqual(fparents[3], 4)
        self.assertEqual(fparents[4], 5)
        self.assertEqual(fparents[5], 5)
        self.assertEqual(fparents[6], 4)
        self.assertEqual(fparents[7], 4)
        self.assertEqual(fparents[8], 4)


if __name__ == '__main__':
    unittest.main()
