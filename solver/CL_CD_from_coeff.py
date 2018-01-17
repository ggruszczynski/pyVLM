import numpy as np

def get_CL_CD_from_coeff(AR, AoA_deg):
    a0 = 2. * np.pi  # dCL/d_alfa in 2D [1/rad]
    e_w = 0.8  # span efficiency factor, range: 0.8 - 1.0

    a = a0 / (1. + a0 / (np.pi * AR * e_w))

    CL_expected_3d = a * np.deg2rad(AoA_deg)
    CD_ind_expected_3d = CL_expected_3d * CL_expected_3d / (np.pi * AR * e_w)

    return CL_expected_3d, CD_ind_expected_3d
