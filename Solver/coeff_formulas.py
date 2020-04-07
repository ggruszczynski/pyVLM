import numpy as np
from scipy import interpolate
import warnings

def get_CL_CD_free_wing(AR, AoA_deg):
    a0 = 2. * np.pi  # dCL/d_alfa in 2D [1/rad]
    e_w = 0.8  # span efficiency factor, range: 0.8 - 1.0

    a = a0 / (1. + a0 / (np.pi * AR * e_w))

    CL_expected_3d = a * np.deg2rad(AoA_deg)
    CD_ind_expected_3d = CL_expected_3d * CL_expected_3d / (np.pi * AR * e_w)

    return CL_expected_3d, CD_ind_expected_3d


def get_CL_CD_submerged_wing(AR, AoA_deg, K=1, c=0, h=0):
    """ 
    :param AR: 
    :param AoA_deg: 
    :param K:  coefficient  accounting for free surface effects
    on steady lift for an uncambered 2D thin foial at an angle of attack.
    
    CL_with_free_surface_effects = CL * K
    CL = 2. * np.pi
    :return: 
    """

    a0 = 2. * np.pi * K  # dCL/d_alfa in 2D [1/rad]
    e_w = 0.8  # span efficiency factor, range: 0.8 - 1.0
    a = a0 / (1. + a0 / (np.pi * AR * e_w))
    CL_expected_3d = a * np.deg2rad(AoA_deg)

    # CL_expected_3d = AR* 2*np.pi*np.deg2rad(AoA_deg)/(2+np.sqrt(AR*AR+4))

    CD_ind_expected_3d = CL_expected_3d * CL_expected_3d / (np.pi * AR * e_w)
    return CL_expected_3d, CD_ind_expected_3d


def calc_free_surface_effect_on_CL(Fn, h_over_chord):
    """ 
    This functions returns coefficient 'K' accounting for free surface effects
    on steady lift for an uncambered 2D thin foial at an angle of attack.
    Source: 
    Hough and Moran 1969 (original)
    "Hydrodynamics of High-Speed Marine Vehicles" Odd M. Faltinsen, chapter 6.8 p 199

    CL_with_free_surface_effects = CL * K
    
    :param Fn: Froude number with h as length parameter
    :param h_over_chord: ratio of foil_submerge/MAC
    :return: 
    """
    # h - foilsubmerge [m]
    # MAC - mean aerodynamic chord [m]


    if (h_over_chord > 1.1 or h_over_chord < 0.9):
        raise ValueError("no data for foil submerge / foil chord other than 1")

    # data below are for hc = h / c = 1
    K = [1, 0.72, 0.6, 0.62, 0.65, 0.76, 0.85, 0.9, 0.91, 0.92]  # [-] 2D free surface lift correction coefficient
    Fnh = [0, 1, 1.5, 2, 2.5, 4, 6, 8, 10, 25]  # Froude number with h as parameter

    if (Fn < 9):
        warnings.warn("To use mirror vortex modeling technique it is recommended to be in high Freud number regime.")
        #  source:
        # "Hydrodynamics of High-Speed Marine Vehicles" Odd M. Faltinsen, chapter 6.8 p 200

    if (Fn > max(Fnh)):
        raise ValueError("Fnh is out of interpolation range  Fnh = %0.2f", Fn)

    fun_handle = interpolate.interp1d(Fnh, K)  # use interpolation function returned by `interp1d`
    K = fun_handle(Fn)

    return K


# import matplotlib.pyplot as plt
# import numpy as np

# xnew = np.linspace(0, 15, num=100)
# ynew = np.array([ calc_CLFreeSurfaceEffect(xnew[i],1) for i in range(len(xnew))])
#
# plt.xlabel('Fn')
# plt.ylabel('K')
# plt.title('Effect of foil submerge on lift')
# plt.plot(xnew, ynew, marker=".", linestyle="-")
# plt.grid(True)
# plt.show()