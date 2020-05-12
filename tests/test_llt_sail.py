import numpy as np
from numpy.testing import assert_almost_equal

from unittest import TestCase

from Solver.mesher import make_panels_from_points
from Solver.winds import Winds
from Solver.llt_sail_opt_solver import calc_circulation_llt
from Solver.vlm_solver import calc_induced_velocity
from Solver.geometry_calc import rotation_matrix


class TestForces(TestCase):
    def test_llt_unconstrained_optimization(self):
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
        # This example follows the case 4.2.1 from GGruszczynski BSc Thesis 'Optimization of an upwind sail geometry in order to maximize thrust.' 2013
        sail_span = 10.  # height of the sail

        # Points defining wing (x,y,z) #
        le_NW = np.array([0, 0., sail_span, ])  # leading edge North - West coordinate
        # mirror in water surface
        le_SW = np.array([0, 0., -sail_span])  # leading edge South - West coordinate

        # make a lifting line instead of panels
        te_NE = le_NW  # trailing edge North - East coordinate
        te_SE = le_SW  # trailing edge South - East coordinate

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
            47.95906141, 50.16407573, 51.75658607, 52.79171779, 53.3017827,
            53.3017827, 52.79171779, 51.75658607, 50.16407573, 47.95906141,
            45.05245163, 41.29808066, 36.43948293, 29.96135264, 20.49987286
        ])

        assert_almost_equal(expected_result, gamma_magnitude)

        V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
        V_app_fs = V_app_infs - V_induced
        spans = np.array([panel.get_panel_span() for panel in panels1D])
        Thrust_inviscid_per_Panel = V_app_fs[:, 1] * gamma_magnitude * spans * rho
        Thrust_inviscid_total = sum(Thrust_inviscid_per_Panel) / 2  # half of the sail is mirrored in the water

        assert_almost_equal(680.4352293328566, Thrust_inviscid_total)
        ### Calculate chord assuming CL

        # Use polar calculated for: NACA 2415, Re = 2e6 by XFoil
        # CL/CD max for AoA = 8 --> CL(8) = 1.0913, CD(8) = 0.01327
        # alfa_0 = -2.0 [deg]
        a = 2 * np.pi  # the slope of the lift coefficient versus angle of attack line is 2*PI units per radian
        CL = 1.0913
        CD = 0.01327
        alfa_0 = -2.0 * np.pi / 180

        alfa_app_fs = np.arctan(V_app_fs[:, 1] / V_app_fs[:, 0])
        alfa_app_infs = np.array(
            [winds.get_app_alfa_infs_at_h(tws_at_ctr_point) for tws_at_ctr_point in tws_at_ctr_points])
        alfa_ind = alfa_app_infs - alfa_app_fs
        phi = alfa_app_fs - alfa_0 - CL / a  # sail_twist eq.2.22 from GG
        phi_deg = np.rad2deg(phi)

        phi_deg_expected = np.array([-4.96381481, -4.96381481, -4.96381481, -4.96381481, -4.96381481,
                                     -4.96381481, -4.96381481, -4.96381481, -4.96381481, -4.96381481,
                                     -4.96381481, -4.96381481, -4.96381481, -4.96381481, -4.96381481,
                                     -4.96381481, -4.96381481, -4.96381481, -4.96381481, -4.96381481])

        assert_almost_equal(phi_deg_expected, phi_deg)

        V_app_fs_length = np.linalg.norm(V_app_fs, axis=1)
        chord = 2 * gamma_magnitude / (CL * V_app_fs_length)  # eq2.21 from GG

        # Total aerodynamic force acting on the sail; section 2.2.2 from GG
        Fa = 0.5 * rho * V_app_fs_length * V_app_fs_length * np.sqrt(CL * CL + CD * CD) * chord
        psi = np.arctan(CD / CL)
        theta = np.pi / 2 + psi - alfa_app_fs
        Thrust_viscid_per_Panel = Fa * np.cos(theta)
        Thrust_viscid_total = sum(Thrust_viscid_per_Panel) / 2  # half of the sail is mirrored in the water

        assert_almost_equal(521.9046999835174, Thrust_viscid_total)
