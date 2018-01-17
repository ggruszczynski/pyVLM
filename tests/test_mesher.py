import numpy as np
from numpy.testing import assert_almost_equal

from solver.panel import Panel
from solver.mesher import \
    make_panels_from_points, \
    discrete_segment, \
    make_point_mesh
from unittest import TestCase


class TestMesher(TestCase):
    def setUp(self):
        # GEOMETRY DEFINITION #
        # it is assumed that the freetream velocity is V [vx,0,vz], where vx > 0
        # Parameters
        self.c_root = 4.0  # root chord length
        self.c_tip = 2.0  # tip chord length
        self.wing_span = 16  # wing span length

        # Points defining wing
        self.le_sw = np.array([0, 0, 0])
        self.le_nw = np.array([self.wing_span, 0, 0])

        self.te_se = np.array([0, self.c_root, 0])
        self.te_ne = np.array([self.wing_span, self.c_tip, 0])

        # MESH DENSITY
        self.ns = 10  # number of panels spanwise
        self.nc = 5  # number of panels chordwise

    def test_make_discrete_segment(self):
        line = discrete_segment(self.le_sw, self.te_se, self.nc)

        expected_line = np.array(
            [[0., 0., 0.],
             [0., 0.8, 0.],
             [0., 1.6, 0.],
             [0., 2.4, 0.],
             [0., 3.2, 0.],
             [0., 4., 0.]])

        assert np.allclose(line, expected_line)

    def test_make_point_mesh(self):
        s_line = discrete_segment(self.le_sw, self.te_se, self.nc)
        n_line = discrete_segment(self.le_nw, self.te_ne, self.nc)

        mesh = make_point_mesh(s_line, n_line, self.ns)
        expected_mesh0 = np.array(
            [[[0.0, 0., 0.],
              [1.6, 0., 0.],
              [3.2, 0., 0.],
              [4.8, 0., 0.],
              [6.4, 0., 0.],
              [8.0, 0., 0.],
              [9.6, 0., 0.],
              [11.2, 0., 0.],
              [12.8, 0., 0.],
              [14.4, 0., 0.],
              [16.0, 0., 0.]]])

        assert np.allclose(mesh[0], expected_mesh0)

    def test_make_panels_from_points(self):
        panels, _ = make_panels_from_points(
            [self.le_sw, self.te_se,
             self.le_nw, self.te_ne],
            [self.nc, self.ns])

        expected_panel = Panel(np.array([4.8, 0.68, 0.]),
                               np.array([4.8, 0., 0.]),
                               np.array([6.4, 0., 0.]),
                               np.array([6.4, 0.64, 0.]))

        assert np.allclose(panels[0][3].p1, expected_panel.p1)
        assert np.allclose(panels[0][3].p2, expected_panel.p2)
        assert np.allclose(panels[0][3].p3, expected_panel.p3)
        assert np.allclose(panels[0][3].p4, expected_panel.p4)

    def test_dimensions(self):
        panels, _ = make_panels_from_points(
            [self.le_sw, self.te_se,
             self.le_nw, self.te_ne],
            [self.nc, self.ns])

        rows, cols = panels.shape

        assert cols == self.ns
        assert rows == self.nc
