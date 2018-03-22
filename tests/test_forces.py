
import numpy as np

from solver.vlm_solver import calc_circulation
from solver.mesher import make_panels_from_points
from solver.geometry_calc import rotation_matrix
from solver.CL_CD_from_coeff import get_CL_CD_from_coeff
from solver.forces import calc_force_wrapper, calc_pressure
from solver.vlm_solver import is_no_flux_BC_satisfied, calc_induced_velocity

from numpy.testing import assert_almost_equal

from solver.geometry_calc import rotation_matrix
from unittest import TestCase
from numpy.linalg import norm

class TestForces(TestCase):

    def test_CL(self):
        ### WING DEFINITION ###
        # Parameters #
        chord = 1.  # chord length
        half_wing_span = 10.  # wing span length

        # Points defining wing (x,y,z) #
        le_NW = np.array([0., half_wing_span, 0.])  # leading edge North - West coordinate
        le_SW = np.array([0., -half_wing_span, 0.])  # leading edge South - West coordinate

        te_NE = np.array([chord, half_wing_span, 0.])  # trailing edge North - East coordinate
        te_SE = np.array([chord, -half_wing_span, 0.])  # trailing edge South - East coordinate

        AoA_deg = 3.0  # Angle of attack [deg]
        Ry = rotation_matrix([0, 1, 0], np.deg2rad(AoA_deg))
        # we are going to rotate the geometry

        ### MESH DENSITY ###
        ns = 20  # number of panels (spanwise)
        nc = 3  # number of panels (chordwise)

        panels, mesh = make_panels_from_points(
            [np.dot(Ry, le_SW),
             np.dot(Ry, te_SE),
             np.dot(Ry, le_NW),
             np.dot(Ry, te_NE)],
            [nc, ns])

        rows, cols = panels.shape
        N = rows * cols

        ### FLIGHT CONDITIONS ###
        V = [10.0, 0.0, 0.0]
        V_app_infw = np.array([V for i in range(N)])
        rho = 1.225  # fluid density [kg/m3]

        ### CALCULATIONS ###
        gamma_magnitude, v_ind_coeff = calc_circulation(V_app_infw, panels)
        V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
        V_app_fw = V_app_infw + V_induced

        assert is_no_flux_BC_satisfied(V_app_fw, panels)

        F = calc_force_wrapper(V_app_infw, gamma_magnitude, panels, rho=rho)
        p = calc_pressure(F, panels)

        ### compare vlm with book formulas ###
        # reference values - to compare with book formulas
        AR = 2 * half_wing_span / chord
        S = 2 * half_wing_span * chord
        CL_expected, CD_ind_expected = get_CL_CD_from_coeff(AR, AoA_deg)

        total_F = np.sum(F, axis=0)
        q = 0.5 * rho * (np.linalg.norm(V) ** 2) * S
        CL_vlm = total_F[2] / q
        CD_vlm = total_F[0] / q

        rel_err_CL = abs((CL_expected - CL_vlm)/CL_expected)
        rel_err_CD = abs((CD_ind_expected - CD_vlm) / CD_ind_expected)
        assert rel_err_CL < 0.01
        assert rel_err_CD < 0.18