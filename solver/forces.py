import numpy as np
from solver.vlm_solver import calc_induced_velocity


def calc_force_wrapper(V_app_infw, gamma_magnitude, panels, rho=1):
    from solver.vortices import v_induced_by_semi_infinite_vortex_line, v_induced_by_finite_vortex_line
    """
    force = rho* (V_app_fw_at_cp x gamma)
    :param V: apparent wind finite sail (including all induced velocities) at control point
    :param gamma_magnitude: vector
    :param rho: 
    :return: 
    """

    panels_1d = panels.flatten()
    N = len(panels_1d)
    v_ind_coeff = np.full((N, N, 3), 0., dtype=float)

    for i in range(0, N):
        cp = panels_1d[i].get_cp_position()
        for j in range(0, N):
            # velocity induced at i-th control point by j-th vortex
            v_ind_coeff[i][j] = panels_1d[j].get_horse_shoe_induced_velocity(cp, V_app_infw[j])

    V_induced = calc_induced_velocity(v_ind_coeff, gamma_magnitude)
    V_at_cp = V_app_infw + V_induced

    force = np.full((N, 3), 0., dtype=float)
    for i in range(0, N):
        [A, B, C, D] = panels_1d[i].get_vortex_ring_position()
        bc = C - B
        gamma = bc * gamma_magnitude[i]
        force[i] = rho * np.cross(V_at_cp[i], gamma)

    return force


def calc_pressure(force, panels):
    panels_1d = panels.flatten()

    n = len(panels_1d)
    p = np.zeros(shape=n)

    for i in range(n):
        area = panels_1d[i].get_panel_area()
        n = panels_1d[i].get_normal_to_panel()
        p[i] = np.dot(force[i], n) / area

    return p
