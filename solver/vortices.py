
import numpy as np
from numpy.linalg import norm

def normalize(x):
    xn = x / norm(x)
    return xn

def is_in_vortex_core(tab):
    return norm(tab) < 1e-9

def v_induced_by_semi_infinite_vortex_line(P, A, r0, gamma=1):
    """
    Biot-Savart law,
    Formula from Katz & Plotkin eq 2.69 p39
    v_ind = gamma*(cos β1 − cos β2) /(4*pi*d)   for semi infinite vortex: β2--> pi 

     ^
     |						Induced velocity at point P due
     |	   + P(x,y,z)		to a semi-infinite straight vortex line
     |						starting at A and pointing in the direction of r0
     +---x=============> r0	
         A		 
    
    Parameters
    ----------
    P, A, B : array_like
              P - point of reference
              A - staring point of the vortex
              r0 - vortex directional vector (apparent wind of infininte sail 
              (we dont want to iteratively solve induced wind, thus infinite))
    gamma : circulation

    Returns
    -------
    v : float
    """

    ### works fine but a simpler formula can be used
    # # Area of a trapezoid:
    # # Area = |distance||r0| = |r0 x r1|
    # ap = P-A
    # r0_cross_ap = np.cross(r0, ap)
    # distance = norm(r0_cross_ap)/norm(r0)
    #
    # # calculate induced wind
    # direction = normalize(r0_cross_ap)
    #
    # magnitude = np.dot(r0, ap)/(norm(r0)*norm(ap)) + 1.  #cos(β1) − cos(pi)
    # magnitude *= gamma/(4.*np.pi*distance)
    #
    # v_ind = magnitude*direction
    #

    #formula from "Modern Adaption of Prandtl’s Classic Lifting-Line Theory" by Philips & Snyder

    u_inf = normalize(r0)
    ap = P - A
    norm_ap = norm(ap)

    v_ind = np.cross(u_inf, ap) / (norm_ap *(norm_ap - np.dot(u_inf,ap))) # TODO consider checking is_in_vortex_core
    v_ind *= gamma/(4.*np.pi)
    return v_ind


def v_induced_by_finite_vortex_line(P, A, B, gamma=1):
    """
    Biot-Savart law,
    Formua from Katz & Plotkin eq 2.72 p41
    
    Y^						Induced velocity at point P due
     |	   + P(x,y,z)		to a finite straight line vortex
     |						defined by points A and B.
     +=======+--------> X	Circulation from A --> B.
     A		 B

    Parameters
    ----------
    P, A, B : array_like
              P - point of reference
              A, B - points of the vortex
    gamma : circulation

    Returns
    -------
    v : float
    """
    BA = np.array(B-A)
    PA = np.array(P-A)
    PB = np.array(P-B)

    PA_cross_PB = np.cross(PA, PB)

    if is_in_vortex_core([PA, PB, PA_cross_PB]):
        return [0,0,0]

    else:
        v_ind = PA_cross_PB / np.square(norm(PA_cross_PB))
        v_ind *= np.dot(BA, (normalize(PA) - normalize(PB)))
        v_ind *= gamma / (4*np.pi)

        return v_ind

def v_induced_by_horseshoe_vortex(P, A, B, r0, gamma=1):
    """
    Induced velocity at point P due to a horseshoe vortex
    of strenght gamma=1 spatially positioned by points A and B,
    extended to x_Inf(+) in a 3D euclidean space. Circulation
    direction is: x_Inf(+) -> A -> B -> x_Inf(+)

                ^
              y |                Points defining the horseshoe
    V_inf       |                are named clockwise.
    ->     B----|->--+...>...    A direction vector is
    ->     |    |    |           calculated for each vortex.
    ->     ^    +----|------>
    ->	   |         |       x
    ->	   A----<----+...<...

    Parameters
    ----------
    P, A, B : array_like
              P - point of reference
              A, B - points of the horseshoe vortex
              r0 - vortex directional vector (apparent wind of infininte sail 
              (we dont want to iteratively solve induced wind, thus infinite))

    Returns
    -------
    v : circulation
    """

    vB = v_induced_by_semi_infinite_vortex_line(P, B, r0, gamma=gamma)
    vAB = v_induced_by_finite_vortex_line(P, A, B, gamma=gamma)
    vA = v_induced_by_semi_infinite_vortex_line(P, A, r0, gamma=-1*gamma)

    v = vA + vB + vAB
    return v
