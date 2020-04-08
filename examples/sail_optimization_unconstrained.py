import numpy as np

from Solver.vlm_solver import calc_induced_velocity, is_no_flux_BC_satisfied
from Solver.mesher import make_panels_from_points
from Solver.geometry_calc import rotation_matrix
from Solver.llt_sail_opt_solver import calc_circulation_llt
from Solver.winds import Winds

### GEOMETRY DEFINITION ###

"""
    This example shows how to use the pyVLM class in order
    to generate the wing planform.

    After defining the conditions (airspeed and AOA),
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

### SAIL DEFINITION ###
# This example follows the case 4.2.1 from GGruszczynski BSc Thesis 'Optimization of an upwind sail geometry in order to maximize thrust.' 2013
# Parameters #
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
# winds = Winds(alfa_real_deg=alfa_real_deg,tws_ref=tws_ref, V_yacht=V_yacht, is_flat_profile=False)
panels1D = panels.flatten()
ctr_points = np.array([panel.get_ctr_point_position() for panel in panels1D])
tws_at_ctr_points = np.array([winds.get_true_wind_speed_at_h(abs(ctr_point[2])) for ctr_point in ctr_points])

V_app_infs = np.array([winds.get_app_infs_at_h(tws_at_ctr_point) for tws_at_ctr_point in tws_at_ctr_points])

gamma_magnitude, v_ind_coeff = calc_circulation_llt(V_app_infs, panels)

V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
V_app_fs = V_app_infs - V_induced
spans = np.array([panel.get_panel_span() for panel in panels1D])
Thrust_inviscid_per_Panel = V_app_fs[:, 1] * gamma_magnitude * spans * rho
Thrust_inviscid_total = sum(Thrust_inviscid_per_Panel)/2
print(f"Thrust in the direction of yacht movement (including leeway) without profile drag = {Thrust_inviscid_total} [N]")

### Calculate chord assuming CL

# Use polar calculated for: NACA 2415, Re = 2e6 by XFoil
# CL/CD max for AoA = 8 --> CL(8) = 1.0913, CD(8) = 0.01327
# alfa_0 = -2.0 [deg]
a = 2*np.pi  # the slope of the lift coefficient versus angle of attack line is 2*PI units per radian
CL = 1.0913
CD = 0.01327
alfa_0 = -2.0 * np.pi/180

alfa_app_fs = np.arctan(V_app_fs[:, 1]/V_app_fs[:, 0])
alfa_app_infs = np.array([winds.get_app_alfa_infs_at_h(tws_at_ctr_point) for tws_at_ctr_point in tws_at_ctr_points])
alfa_ind = alfa_app_infs - alfa_app_fs
phi = alfa_app_fs - alfa_0 - CL/a  # sail_twist eq.2.22 from GG
phi_deg = np.rad2deg(phi)
# phi_boom = phi[0] - phi # take the middle one


import matplotlib.pyplot as plt
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

fig_name = f'plots/sample_plot2D_param_ns={ns}.png'

# -------------------- make dummy plot --------------------
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(14, 8))

axes = plt.gca()
upper_half = int(ns/2)
plt.plot(gamma_magnitude[upper_half:], ctr_points[upper_half:, 2],
         color="black", marker="x", markevery=1, markersize=5, linestyle="--", linewidth=2,
         label='gamma_magnitude')

# plt.plot(phi_deg[upper_half:], ctr_points[upper_half:, 2],
#          color="black", marker="x", markevery=1, markersize=5, linestyle="--", linewidth=2,
#          label='phi_deg')

# plt.plot(LA.norm(V_aw_infs, axis=1), ctr_points[:, 2],
#          color="black", marker=">", markevery=1, markersize=5, linestyle="-", linewidth=2,
#          label='V_aw_infs')

# plt.plot(LA.norm(V_induced, axis=1), ctr_points[:, 2],
#          color="black", marker="v", markevery=1, markersize=5, linestyle="-.", linewidth=2,
#          label='V_induced')

# plt.plot(LA.norm(V_app_fs, axis=1), ctr_points[:, 2],
#          color="black", marker="<", markevery=1, markersize=5, linestyle=":", linewidth=2,
#          label='V_app_fs')

# ------ format y axis ------ #
# yll = ctr_points[upper_half:, 2].min()
# yhl = ctr_points[upper_half:, 2].max()
yll = 0
yhl = 10
axes.set_ylim([yll, yhl])
axes.set_yticks(np.linspace(yll, yhl, 11))
# axes.set_yticks(np.arange(yll, yhl, 1E-2))
# axes.set_yticks([1E-4, 1E-6, 1E-8, 1E-10, 1E-12])
# axes.yaxis.set_major_formatter(xfmt)

# plt.yscale('log')


# ------ format x axis ------ #
# plt.xlim(x1-0.5, x2+0.5)

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # scilimits=(-8, 8)


plt.title(f'Sample plot')
# plt.xlabel(r'$\Gamma_{opt}$')
plt.xlabel(r'$Stuff$')
plt.ylabel(r'$Height$')
plt.legend()
plt.grid()

fig = plt.gcf()  # get current figure
fig.savefig(fig_name, bbox_inches='tight')
plt.show()
# plt.close(fig)  # close the figure
print("bye")

