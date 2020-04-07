import numpy as np

from Solver.vortices import \
    v_induced_by_semi_infinite_vortex_line, \
    v_induced_by_finite_vortex_line, \
    v_induced_by_horseshoe_vortex, \
    normalize



class Panel(object):
    """
           y ^
             |              Each panel is defined by the (x, y) coordinates
        P3-C-|-D-P4         of four points - namely P1, P2, P3 and P4 -
         | | | |  |         ordered clockwise. Points defining the horseshoe
         | | +-P--|--->     - A, B, C and D - are named clockwise as well.
         | |   |  |   x
        P2-B---A-P1

    Parameters
    ----------
    P1, P2, P3, P4 : array_like
                     Corner points in a 3D euclidean space
    """
    panel_counter = 0
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        self.counter = Panel.panel_counter
        Panel.panel_counter += 1

        self._check_in_plane()

    def _check_in_plane(self):
        P1P2 = self.p1 - self.p2
        P3P4 = self.p4 - self.p3

        vec = np.cross(P1P2, P3P4)
        if np.linalg.norm(vec) > 1e-12:
            raise ValueError("Points on Panel are not on the same plane!")


    def get_normal_to_panel(self):
        p1_p2 = self.p2 - self.p1
        p1_p4 = self.p4 - self.p1

        # n = np.cross(p1_p2, p1_p4)
        n = np.cross(p1_p4, p1_p2)
        n = normalize(n)
        return n

    def get_panel_area(self):
        p = [self.p1, self.p2, self.p3, self.p4]

        path = []
        for i in range(len(p) - 1):
            step = p[i + 1] - p[i]
            path.append(step)

        # path.append(p[3] - p[0])
        # path.append(p[0] - p[1])

        area = 0
        for i in range(len(path) - 1):
            s = np.cross(path[i], path[i + 1])
            s = np.linalg.norm(s)
            area += 0.5 * s

        return area

    def get_panel_span(self):
        [_, B, C, _] = self.get_vortex_ring_position()
        BC = C-B

        dl = 0
        for i in range(3):
            dl += BC[i]*BC[i]
        span = np.sqrt(dl)
        return span


    def get_ctr_point_position(self):
        """
         For a given panel defined by points P1, P2, P3 and P4
         returns the position of the control point P.

                  ^
                 y|                Points defining the panel
                  |                are named clockwise.
          P3------|------P4
           |      |      |
           |      |      |
           |      +-P----------->
           |             |      x
           |             |
          P2-------------P1

         Parameters
         ----------
         P1, P2, P3, P4 : array_like
                          Points that define the panel

         Returns
         -------
             P - control point where the boundary condition V*n = 0
                 is applied according to the Vortice Lattice Method.
         """
        p2_p1 = self.p1 - self.p2
        p1_p4 = self.p4 - self.p1
        ctr_p = self.p2 + p2_p1 * (3. / 4.) + p1_p4 / 2.
        return ctr_p

    def get_cp_position(self):
        """
         For a given panel defined by points P1, P2, P3 and P4
         returns the position of the centre of pressure

                  ^
                 y|                Points defining the panel
                  |                are named clockwise.
          P3------|------P4
           |      |      |
           |      |      |
           |      +-P----------->
           |             |      x
           |             |
          P2-------------P1

         Parameters
         ----------
         P1, P2, P3, P4 : array_like
                          Points that define the panel

         Returns
         -------
         results : dict
             CP - centre of pressure, when calculating CL, CD it assumed that the force is attached to this point
             The induced wind is calculated at CP, and then U_inf + U_ind is used to find the force.
         """
        p2_p1 = self.p1 - self.p2
        p1_p4 = self.p4 - self.p1
        cp = self.p2 + p2_p1 * (1. / 4.) + p1_p4 / 2.
        return cp

    def get_vortex_ring_position(self):
        """
        For a given panel defined by points P1, P2, P3 and P4
        returns the position of the horseshoe vortex defined
        by points A, B and its control point P.

                  ^
                 y|                Points defining the panel
                  |                are named clockwise.
         P3--C----|---P4---D
          |  |    |    |   |
          |  |    |    |   |
          |  |    +----|---------->
          |  |         |   |      x
          |  |         |   |
         P2--B---------P1--A

        Parameters
        ----------
        P1, P2, P3, P4 : array_like
                         Points that define the panel

        Returns
        -------
        results : dict
            A, B, C, D - points that define the vortex ring
        """

        p2_p1 = self.p1 - self.p2
        p3_p4 = self.p4 - self.p3

        A = self.p1 + p2_p1 / 4.
        B = self.p2 + p2_p1 / 4.
        C = self.p3 + p3_p4 / 4.
        D = self.p4 + p3_p4 / 4.

        return [A, B, C, D]


    def get_vortex_ring_induced_velocity(self):
        ctr_p = self.get_ctr_point_position()
        [A, B, C, D] = self.get_vortex_ring_position()

        v_AB = v_induced_by_finite_vortex_line(ctr_p, A, B)
        v_BC = v_induced_by_finite_vortex_line(ctr_p, B, C)
        v_CD = v_induced_by_finite_vortex_line(ctr_p, C, D)
        v_DA = v_induced_by_finite_vortex_line(ctr_p, D, A)

        v = v_AB + v_BC + v_CD + v_DA
        return v

    def get_horse_shoe_induced_velocity(self, ctr_p, V_app_infw):
        [A, B, C, D] = self.get_vortex_ring_position()
        v = v_induced_by_horseshoe_vortex(ctr_p, B, C, V_app_infw)
        return v

