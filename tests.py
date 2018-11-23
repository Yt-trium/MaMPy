import algo2 as mtlib
import numpy as np
import unittest

class TestFindPixelParentFunctions(unittest.TestCase):
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

        with self.assertRaises(IndexError):
            mtlib.find_pixel_parent(self.parents, 42)

    def test_negative_pixel_index(self):
        """We expect it to fail."""

        # Warning: in numpy you can fetch in both ways.
        self.assertEqual(
            mtlib.find_pixel_parent(self.parents, -3),
            mtlib.find_pixel_parent(self.parents, len(self.parents) - 3))
        self.assertEqual(
            mtlib.find_pixel_parent(self.parents, -1),
            mtlib.find_pixel_parent(self.parents, len(self.parents) -1))

    def test_wrong_parents_data(self):
        """The parents image should contains pixel valid indices"""

        wrong_parents = np.array([0, 3, 2, 10, 49, 50])

        # We expect it to fail.
        with self.assertRaises(IndexError):
            mtlib.find_pixel_parent(wrong_parents, 3)

    def test_parents(self):
        self.assertEqual(mtlib.find_pixel_parent(self.parents, 1), 1)
        self.assertEqual(mtlib.find_pixel_parent(self.parents, 3), 1)
        self.assertEqual(mtlib.find_pixel_parent(self.parents, 0), 1)
        self.assertEqual(mtlib.find_pixel_parent(self.parents, 2), 1)
        self.assertEqual(mtlib.find_pixel_parent(self.parents, 4), 1)
        self.assertEqual(mtlib.find_pixel_parent(self.parents, 5), 1)

if __name__ == '__main__':
    unittest.main()
