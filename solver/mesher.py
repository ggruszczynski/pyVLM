import numpy as np
from solver.panel import Panel


def make_panels_from_points(points, grid_size):
    """
    this is the main meshing method
    :param points: 
    :param grid_size: 
    :return: 
    """
    le_SW, te_SE, le_NW, te_NE = points
    nc, ns = grid_size
    south_line = discrete_segment(le_SW, te_SE, nc)
    north_line = discrete_segment(le_NW, te_NE, nc)

    mesh = make_point_mesh(south_line, north_line, ns)
    panels = make_panels_from_mesh(mesh)
    return panels, mesh

def discrete_segment(p1, p2, n):
    segment = []
    step = (p2-p1)/n

    for i in range(n):
        point = p1 + i * step
        segment.append(point)

    segment.append(p2)
    return np.array(segment)

def make_point_mesh(segment1, segment2, n):
    mesh = []
    for p1,p2 in zip(segment1,segment2):
        s =  discrete_segment(p1,p2,n)
        mesh.append(np.array(s))

    return np.array(mesh)

def make_panels_from_mesh(mesh):
    panels = []

    n_lines = mesh.shape[0]
    n_points_per_line = mesh.shape[1]

    for i in range(n_lines-1):
        panels.append([])
        for j in range(n_points_per_line-1):
            pSW = mesh[i][j]
            pSE = mesh[i+1][j]
            pNW = mesh[i][j+1]
            pNE = mesh[i+1][j+1]
            panel = Panel(p1 = pSE,
                          p2 = pSW,
                          p4 = pNE,
                          p3 = pNW)
            panels[i].append(panel)

    return np.array(panels)
