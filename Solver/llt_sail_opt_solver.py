import numpy as np


def assembly_sys_of_eq_llt(V_app_infs, panels):
    panels1D = panels.flatten()
    N = len(panels1D)

    A = np.zeros(shape=(N, N))  # Aerodynamic Influence Coefficient matrix
    RHS = np.zeros(shape=N)
    v_ind_coeff = np.full((N, N, 3), 0., dtype=float)
    ctr_points = np.array([panel.get_ctr_point_position() for panel in panels.flatten()])

    for i in range(0, N):
        RHS[i] = V_app_infs[i][1] * panels1D[i].get_panel_span()
        for j in range(0, N):
            # velocity induced at i-th control point by j-th vortex
            v_ind_coeff[i][j] = panels1D[j].get_horse_shoe_induced_velocity(ctr_points[i], V_app_infs[j])

    for i in range(0, N):
        for j in range(0, N):
            A[i][j] = (v_ind_coeff[i][j][1] * panels1D[j].get_panel_span() \
                       + v_ind_coeff[j][i][1] * panels1D[i].get_panel_span())

    return A, RHS, v_ind_coeff  # np.array(v_ind_coeff)


def calc_circulation_llt(V_app_infs, panels):
    # it is assumed that the freestream velocity is V [vx,vy,0], where vx > 0  # TODO: ensure that it is right - different csys in case of rectangular wing?

    A, RHS, v_ind_coeff = assembly_sys_of_eq_llt(V_app_infs, panels)
    gamma_magnitude = np.linalg.solve(A, RHS)

    return gamma_magnitude, v_ind_coeff
