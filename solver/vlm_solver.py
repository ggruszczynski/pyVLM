import numpy as np


def assembly_sys_of_eq(V_app_infw, panels):
    rows, cols = panels.shape
    N = rows * cols
    A = np.zeros(shape=(N, N))  # Aerodynamic Influence Coefficient matrix
    RHS = np.zeros(shape=N)

    panels1D = panels.flatten()
    v_ind_coeff = np.full((N, N, 3), 0., dtype=float)

    for i in range(0, N):
        panel_surf_normal = panels1D[i].get_normal_to_panel()
        ctr_p = panels1D[i].get_ctr_point_postion()
        RHS[i] = -np.dot(V_app_infw[i], panel_surf_normal)

        for j in range(0, N):
                # velocity induced at i-th control point by j-th vortex
                v_ind_coeff[i][j] = panels1D[j].get_horse_shoe_induced_velocity(ctr_p, V_app_infw[j])
                A[i][j] = np.dot(v_ind_coeff[i][j], panel_surf_normal)

    return A, RHS, v_ind_coeff  # np.array(v_ind_coeff)


def calc_circulation(V_app_ifnw, panels):
    # it is assumed that the freestream velocity is V [vx,0,vz], where vx > 0

    A, RHS, v_ind_coeff = assembly_sys_of_eq(V_app_ifnw, panels)
    gamma_magnitude = np.linalg.solve(A, RHS)

    return gamma_magnitude, v_ind_coeff


def calc_induced_velocity(v_ind_coeff, gamma_magnitude):
    N = len(gamma_magnitude)
    V_induced = np.full((N, 3), 0., dtype=float)
    for i in range(N):
        for j in range(N):
            V_induced[i] += v_ind_coeff[i][j] * gamma_magnitude[j]

    return V_induced


def is_no_flux_BC_satisfied(V_app_fw, panels):
    rows, cols = panels.shape
    N = rows * cols
    flux_through_panel = np.zeros(shape=N)

    panels1D = panels.flatten()
    for i in range(0, N):
        panel_surf_normal = panels1D[i].get_normal_to_panel()
        flux_through_panel[i] = -np.dot(V_app_fw[i], panel_surf_normal)

    for flux in flux_through_panel:
        if abs(flux) > 1E-12:
            raise ValueError("Solution error, there is a significant flow through panel!")

    return True