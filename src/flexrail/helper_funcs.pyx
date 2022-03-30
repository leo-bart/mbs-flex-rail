#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:48:51 2022

Auxiliary mathematical funcions

@author: leonardo
"""

import numpy as np
import cython

@cython.boundscheck(False)


cpdef unitaryVector(double [:] vec):
    '''
    Returns an unit vector (norm = 1) in the same direction of the input vector
    
    PARAMETERS
    ----------
    vec: double array
        input vector
        
    RETURNS
    -------
    array, double
        the unitary vector and the original vector norm
    '''
    
    cdef Py_ssize_t i
    cdef double norma = 0
    cdef double[:] unitVec = np.zeros(3)
    
    for i in range(3):
        norma += vec[i] * vec[i]
        
        
    if norma != 0:
        norma = np.sqrt(norma)
        
        for i in range(3):
            unitVec[i] = vec[i] / norma
    else:
        norma = 0.0
        
        for i in range(3):
            unitVec[i] = 0.0
        
    return np.array(unitVec), norma




cpdef singleRotationMatrix(double angleRad, Py_ssize_t order):
    '''
    singleRotationMatrix
    
    Returns the 3x3 rotation matrix around the axis specified by order.
    
    Parameters
    ----------
    angleRad: double
        The angle value in radians
    order: long integer
        The axis of the rotation.
        
    Returns
    -------
    R: array of doubles (C style)
        A 3x3 array with the rotation matrix
    '''
    cdef double [3][3] R
    cdef double ca, sa
    cdef list line
    cdef Py_ssize_t i, j, k
    
    for i in range(3):
        for j in range(3):
            if i == j:
                R[i][j] = 1.
            else:
                R[i][j] = 0.
    
    if angleRad != 0:
        
        
        ca = np.cos(angleRad)
        sa = np.sin(angleRad)
                
        line = [ca,-sa,sa,ca]
        
        k = 0
        for i in range(3):
            if i != order:
                for j in range(3):
                    if j != order:
                        R[i][j] = line[k]
                        k += 1
                
    return R

cpdef singleRotationMatrixDerivative(double angleRad, Py_ssize_t order):
    '''
    singleRotationMatrixDarivative
    
    Returns the dreivative of the 3x3 rotation matrix around the axis specified by order.
    
    Parameters
    ----------
    angleRad: double
        The angle value in radians
    order: long integer
        The axis of the rotation.
        
    Returns
    -------
    R: array of doubles (C style)
        A 3x3 array with the rotation matrix derivative with respect to order
    '''
    cdef double [3][3] R
    cdef double ca, sa
    cdef list line
    cdef Py_ssize_t i, j, k
    
    for i in range(3):
        for j in range(3):
            R[i][j] = 0.
        
    ca = np.cos(angleRad)
    sa = np.sin(angleRad)
            
    line = [-sa,-ca,ca,-sa]
    
    k = 0
    for i in range(3):
        if i != order:
            for j in range(3):
                if j != order:
                    R[i][j] = line[k]
                    k += 1
                
    return R




cpdef cardanRotationMatrix(double[:] anglesInRad):
    '''
    rotationMatrix(anglesInRad)
    
    Creates a rotation matrix given a vector angleInRad containing the three
    rotations in the order 1,2,3 (Cardan angles)

    Parameters
    ----------
    anglesInRad : array
        Angles of rotation given in rad following Cardan's convention.


    Returns
    -------
    The rotation matrix.

    '''    
    cdef double a,b,c
    
    a = anglesInRad[0]
    b = anglesInRad[1]
    c = anglesInRad[2]
    
    if b == c == 0.:
        # rotation around x
        return np.array(singleRotationMatrix(a,0))
    elif a == b == 0.:
        # rotation around z
        return np.array(singleRotationMatrix(c,2))
    elif a == c == 0.:
        # rotation around y
        return np.array(singleRotationMatrix(b,1))
    elif a == 0.:
        # rotation around y and z
        cb = np.cos(b)
        sb = np.sin(b)
        cc = np.cos(c)
        sc = np.sin(c)
        return np.array([
            [cb*cc, -sc, -sb*cc],
            [sc*cb,  cc, -sb*sc],
            [sb,    0.0, cb    ]
            ])
    elif b == 0.:
        # rotation around x and z
        ca = np.cos(a)
        sa = np.sin(a)
        cc = np.cos(c)
        sc = np.sin(c)
        return np.array([
            [cc, -sc*ca,  sa*sc],
            [sc,  ca*cc, -sa*cc],
            [0.0,     sa,     ca]
            ])
    elif c == 0.:
        # rotation around x and y
        ca = np.cos(a)
        sa = np.sin(a)
        cb = np.cos(b)
        sb = np.sin(b)
        return np.array([
            [cb, -sa*sb, -sb*ca],
            [0.,     ca,    -sa],
            [sb,  sa*cb,  ca*cb]
            ])
    else:
        # all rotations
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)
        return np.array([
            [cb*cc, -sa*sb*cc - sb*ca,  sa*sb - sb*ca*cc],
            [sb*cb, -sa*sb*sb + ca*cc, -sa*cc - sb*sb*ca],
            [   sb,             sa*cb,             ca*cb]
            ])


cpdef cardanRotationMatrixDerivative(double[:] anglesInRad, long derAxis):
    '''
    rotationMatrix(anglesInRad)
    
    Creates a rotation matrix given a vector angleInRad containing the three
    rotations in the order 1,2,3 (Cardan angles)

    Parameters
    ----------
    anglesInRad : array
        Angles of rotation given in rad following Cardan's convention.
    derAxis: long integer
        The number (0,1, or 2) that the defines the axis of the derivative
        
            dR
        ----------
        da_derAXis


    Returns
    -------
    The rotation matrix.

    '''    
    
    cdef double a,b,c
    cdef Py_ssize_t i 
    cdef list R = []
    
    for i in range(3):
        if i == derAxis:
            R.append(np.array(singleRotationMatrixDerivative(anglesInRad[i],i)))
        else:
            R.append(np.array(singleRotationMatrix(anglesInRad[i],i)))
            
    return R[2].dot(R[1].dot(R[0]))
            
        
if __name__ == "__main__":

    v = np.zeros(3)
    u = unitaryVector(v)
        