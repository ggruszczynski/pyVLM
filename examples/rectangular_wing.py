import numpy as np

from solver.vlm_solver import calc_circulation
from solver.mesher import make_panels_from_points
from solver.geometry_calc import rotation_matrix
from solver.CL_CD_from_coeff import get_CL_CD_from_coeff
from solver.forces import calc_force_wrapper, calc_pressure
from solver.vlm_solver import is_no_flux_BC_satisfied, calc_induced_velocity

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

# reference values - only to compare with book formulas
AR = 2 * half_wing_span / chord  # TODO allow tapered wings AR in book formulas
S = 2 * half_wing_span * chord  # TODO allow tapered wings S in book formulas

### MESH DENSITY ###
ns = 15  # number of panels (spanwise)
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

assert is_no_flux_BC_satisfied(V_app_fw, gamma_magnitude, panels, v_ind_coeff)

F = calc_force_wrapper(V_app_infw, gamma_magnitude, panels, rho=rho)
p = calc_pressure(F, panels)

print("gamma_magnitude: \n")
print(gamma_magnitude)
print("DONE")

### compare vlm with book formulas ###
CL_expected, CD_ind_expected = get_CL_CD_from_coeff(AR, AoA_deg)

total_F = np.sum(F, axis=0)
q = 0.5 * rho * (np.linalg.norm(V) ** 2) * S
CL_vlm = total_F[2] / q
CD_vlm = total_F[0] / q

print("\nCL_expected %f \t CD_ind_expected %f" % (CL_expected, CD_ind_expected))
print("CL_vlm %f \t CD_vlm %f" % (CL_vlm, CD_vlm))
print("\n\ntotal_F %s" % str(total_F))
print("=== END ===")
