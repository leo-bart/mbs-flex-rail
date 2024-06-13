# -*- coding: utf-8 -*-
# distutils: language=c++
"""
Created on Thu May  4 23:32:02 2023

@author: lbaru
"""

import numpy as np
cimport numpy as np

def find_convex_hull(double[:, ::1] points):
    cdef int n = points.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] sorted_points = np.empty(n, dtype=np.int_)
    for i in range(n):
        sorted_points[i] = i
    cdef int[: ] lower_hull = [sorted_points[0]]
    cdef int[: ] upper_hull = [sorted_points[0]]
    cdef int point
    for i in range(1, n):
        point = sorted_points[i]
        while len(lower_hull) > 1 and np.cross(
            points[lower_hull[-1]] - points[lower_hull[-2]],
            points[point] - points[lower_hull[-2]]
        ) <= 0:
            lower_hull.pop()
        lower_hull.append(point)
        while len(upper_hull) > 1 and np.cross(
            points[upper_hull[-1]] - points[upper_hull[-2]],
            points[point] - points[upper_hull[-2]]
        ) >= 0:
            upper_hull.pop()
        upper_hull.append(point)
    cdef int[: ] convex_hull = np.concatenate(
        (lower_hull, upper_hull[-2:0:-1])
    )
    return convex_hull
