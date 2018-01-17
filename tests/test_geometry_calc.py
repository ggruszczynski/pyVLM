


import numpy as np
from numpy.testing import assert_almost_equal

from solver.geometry_calc import rotation_matrix
from unittest import TestCase

class TestPanels(TestCase):

    def test_rotations(self):

        Ry = rotation_matrix([0, 1, 0], np.deg2rad(45))
        result = np.dot(Ry, [1, 456, 1])

        expected_result = [1.41421356, 456, 0]

        assert_almost_equal(expected_result,result)