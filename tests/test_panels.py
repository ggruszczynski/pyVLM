"""
    Unit tests of the Panel class and its methods

"""

import numpy as np
from numpy.testing import assert_almost_equal

from Solver.panel import Panel
from unittest import TestCase


class TestPanels(TestCase):
    def setUp(self):
        self.points = [np.array([10, 0, 0]), np.array([0, 0, 0]),
                       np.array([0, 10, 0]), np.array([10, 10, 0])]

        self.panel = Panel(*self.points)

    def test_area(self):
        calculated_area = self.panel.get_panel_area()
        expected_area = 100.0

        assert_almost_equal(calculated_area, expected_area)

    def test_pressure(self):
        from Solver.forces import calc_pressure

        p = calc_pressure(force=np.array([[0, 0, 1], [3, 2, 1]]),
                          panels=np.array([self.panel, self.panel]))

        assert_almost_equal(p, [0.01, 0.01])

    def test_get_ctr_point_postion(self):
        ctr_point = self.panel.get_ctr_point_position()
        expected_ctr_point = [7.5, 5, 0]

        assert_almost_equal(expected_ctr_point, ctr_point)

    def test_get_cp_postion(self):
        cp = self.panel.get_cp_position()
        expected_ctr_point = [2.5, 5, 0]

        assert_almost_equal(expected_ctr_point, cp)

    def test_get_vortex_ring_position(self):
        vortex_ring_position = self.panel.get_vortex_ring_position()
        expected_vortex_riing_position = [[12.5, 0., 0.],
                                          [2.5, 0., 0.],
                                          [2.5, 10., 0.],
                                          [12.5, 10., 0.]]

        assert_almost_equal(expected_vortex_riing_position, vortex_ring_position)

    def test_get_vortex_ring_induced_velocity(self):
        v_ind = self.panel.get_vortex_ring_induced_velocity()
        v_ind_expected = [0, 0, -0.09003163161571061]

        assert_almost_equal(v_ind, v_ind_expected)

    def test_panel_is_not_plane(self):
        with self.assertRaises(ValueError) as context:
            points = [np.array([10, 0, 0]), np.array([0, 666, 0]),
                      np.array([0, 10, 0]), np.array([10, 10, 0])]

            panel = Panel(*points)

        self.assertTrue("Points on Panel are not on the same plane!" in context.exception.args[0])
