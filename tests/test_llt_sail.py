import numpy as np
from numpy.testing import assert_almost_equal

from unittest import TestCase

from Solver.mesher import make_panels_from_points
from Solver.winds import Winds
from Solver.llt_sail_opt_solver import calc_circulation_llt
from Solver.geometry_calc import rotation_matrix


class TestForces(TestCase):
    def test_CL(self):
        ### GEOMETRY DEFINITION ###

        """
            This example shows how to use the pyVLM class in order
            to generate the wing planform.

            After defining the flight conditions (airspeed and AOA),
            the geometry will be characterised using the following
            nomenclature:

            Y  ^    le_NW +--+ te_NE
               |         /    \
               |        /      \
               |       /        \
               +------/----------\---------------->
               |     /            \               X
               |    /              \
             le_SW +-----------------+ te_SE


        """

        ### WING DEFINITION ###
        # Parameters #
        # This example follows the case 4.2.1 from GGruszczynski BSc Thesis
        chord = 0  # chord length - in the LLT theory the ctr points lay on the lifting line
        half_wing_span = 10.  # half wing span length

        # Points defining wing (x,y,z) #
        le_NW = np.array([0, 0., half_wing_span, ])  # leading edge North - West coordinate
        le_SW = np.array([0, 0., -half_wing_span])  # leading edge South - West coordinate
        te_NE = np.array([chord, 0, half_wing_span])  # trailing edge North - East coordinate
        te_SE = np.array([chord, 0., -half_wing_span])  # trailing edge South - East coordinate

        AoA_deg = 0.0  # Angle of attack [deg]
        Ry = rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))

        ### MESH DENSITY ###
        ns = 20  # number of panels (spanwise)
        nc = 1  # number of panels (chordwise)

        panels, mesh = make_panels_from_points(
            [np.dot(Ry, le_SW),
             np.dot(Ry, te_SE),
             np.dot(Ry, le_NW),
             np.dot(Ry, te_NE)],
            [nc, ns])

        ### FLIGHT CONDITIONS ###
        tws_ref = 10  # Free stream of true wind having velocity [m/s] at height z = 10 [m]
        V_yacht = 15  # [m/s]
        alfa_real_deg = 15  # [deg] angle between true wind and direction of boat movement (including leeway)

        rho = 1.225  # fluid density [kg/m3]

        ### CALCULATIONS ###
        winds = Winds(alfa_real_deg=alfa_real_deg, tws_ref=tws_ref, V_yacht=V_yacht, is_flat_profile=True)

        panels1D = panels.flatten()
        ctr_points = np.array([panel.get_ctr_point_position() for panel in panels1D])
        tws_at_ctr_points = np.array([winds.get_true_wind_speed_at_h(abs(ctr_point[2])) for ctr_point in ctr_points])

        V_app_infs = np.array([winds.get_app_infs_at_h(tws_at_ctr_point) for tws_at_ctr_point in tws_at_ctr_points])

        gamma_magnitude, v_ind_coeff = calc_circulation_llt(V_app_infs, panels)

        expected_result = np.array([
            20.49987286, 29.96135264, 36.43948293, 41.29808066, 45.05245163,
            47.95906141, 50.16407573, 51.75658607, 52.79171779, 53.3017827 ,
            53.3017827 , 52.79171779, 51.75658607, 50.16407573, 47.95906141,
            45.05245163, 41.29808066, 36.43948293, 29.96135264, 20.49987286
        ])

        assert_almost_equal(expected_result, gamma_magnitude)
