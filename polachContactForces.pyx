#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Implementation of Polach simplified contact algorithm. Converted from the
FORTRAN original in 

POLACH, O. A Fast Wheel-Rail Forces Calculation Computer Code. 
Vehicle System Dynamics, v. 33, n. sup1, p. 728–739, 1 jan. 1999.

Created on Sun 2024.06.09

@author: Leonardo Baruffaldi
"""


cimport cython
import numpy as np
cimport numpy as np

# Define types for input parameters
@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing for performance
def polach(double Q, double F, double A, double B, double SX, double SY, 
        double OM, double C1, double C2, double C3, bint verb = False):
    """
    FX - longitudinal force in wheel-rail contact Fx
    FY - lateral force in wheel-rail contact Fy
    SX - longitudinal creep sx
    SY - lateral creep sy
    OM - spin ψ
    Q - wheel load Q
    F - coefficient of friction f
    A - semiaxis a of the contact ellipse (in longitudinal direction)
    B - semiaxis b of the contact ellipse (in lateral direction)
    G - modulus of rigidity G
    C1- Kalker’s constant c11
    C2- Kalker’s constant c22
    C3- Kalker’s constant c23

    Parameters
    ----------
    double Q : TYPE
        DESCRIPTION.
    double F : TYPE
        DESCRIPTION.
    double A : TYPE
        DESCRIPTION.
    double B : TYPE
        DESCRIPTION.
    double SX : TYPE
        DESCRIPTION.
    double SY : TYPE
        DESCRIPTION.
    double OM : TYPE
        DESCRIPTION.
    double C1 : TYPE
        DESCRIPTION.
    double C2 : TYPE
        DESCRIPTION.
    double C3 : TYPE
        DESCRIPTION.
    bint verb : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    FX : TYPE
        DESCRIPTION.
    FY : TYPE
        DESCRIPTION.

    """
    cdef double G = 79e12
    cdef double PI = np.pi

    cdef double FX = 0
    cdef double FY = 0
    cdef double MI = 0
    cdef double SYC = SY

    if abs(SY + OM * A) <= abs(SYC):
        SYC = SY + OM * A

    cdef double SC = (SX**2 + SYC**2)**0.5
    if SC == 0:
        return FX, FY

    cdef double S = (SX**2 + SY**2)**0.5
    cdef double CJ = 0
    if S != 0:
        CJ = ((C1 * SX / S)**2 + (C2 * SY / S)**2)**0.5

    cdef double EP = PI * G * A * B * CJ * SC / (4 * Q * F)
    MI = (EP / (1 + EP**2) + np.arctan(EP)) * 2 * F / PI

    cdef double KS = 1 + 6.3 * (1 - np.exp(-A / B))
    cdef double EPM = 8 * B * (A * B)**0.5 * G * C3 * abs(SYC) / (3 * KS * Q * F)
    cdef double DE = (EPM**2 - 1) / (EPM**2 + 1)

    cdef double FYS = 9 * A * F * KS * (EPM * (-DE**3 / 3 + DE**2 / 2 - 1 / 6) + 1 / 3 * (1 - DE**2)**1.5) / 16

    FX = -Q * MI * SX / SC
    FY = -Q * (MI * SY + FYS * OM) / SC

    if verb: print(FX, FY)
    
    return FX, FY