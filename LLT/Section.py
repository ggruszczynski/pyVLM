import numpy as np


class Section:
    x = 0
    y = 0
    z = 0  # section height
    z_c = 0  # height of control points [m]
    b = 0  # b[j] - span of j-th sail section [m],  at j-th control point

    c = 0  # c[j] -  chord of j-th sail section [m], at j-th control point
    phi = 0  # phi[j] - angle of sail twist at j-th control point
    phi_boom = 0  # phi_norm[j] angle of sail twist relative to boom, at j-th control point
    CL = 0  # CL[j] design lift coefficient, which is defined at j-th control point
    CD = 0  # CD[j] drag coefficient corresponding to CL[j], which is defined at j-th control point
    alfa_0 = 0  # Angle of attack for which CL = 0, use [rad]

    # // Use polar calculated for: NACA 2415, Re = 2e6 by XFoil
    # // CL / CD  max for AoA = 8 --> CL(8) = 1.0913, CD(8) = 0.01327
    # // alfa_0 = -2.0[deg]

    def __init__(self):
        pass
