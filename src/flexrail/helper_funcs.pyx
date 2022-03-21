#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:48:51 2022

Auxiliary mathematical funcions

@author: leonardo
"""

import numpy as np

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
    cdef double[:,:] R = np.diag([1.,1.,1.])
    cdef double a,b,c
    
    a = anglesInRad[0]
    b = anglesInRad[1]
    c = anglesInRad[2]
    
    if b == c == 0:
        # rotation around x
        R[1,1] = R[2,2] = np.cos(a)
        R[1,2] = -np.sin(a)
        R[2,1] = -R[1,2]
        return np.array(R)
    elif a == b == 0:
        # rotation around z
        R[0,0] = R[1,1] = np.cos(c)
        R[0,1] = -np.sin(c)
        R[1,0] = -R[0,1]
        return np.array(R)
    elif a == c == 0:
        # rotation around y
        R[0,0] = R[2,2] = np.cos(b)
        R[0,2] = - np.sin(b)
        R[2,0] = -R[0,2]
        return np.array(R)
    elif a == 0:
        # rotation around y and z
        cb = np.cos(b)
        sb = np.sin(b)
        cc = np.cos(c)
        sc = np.sin(c)
        R = np.array([
            [cb*cc, -sc, -sb*cc],
            [sc*cb,  cc, -sb*sc],
            [sb,      0, cb    ]
            ])
        return np.array(R)
    elif b == 0:
        # rotation around x and z
        ca = np.cos(a)
        sa = np.sin(a)
        cc = np.cos(c)
        sc = np.sin(c)
        R = np.array([
            [cc, -sc*ca,  sa*sc],
            [sc,  ca*cc, -sa*cc],
            [ 0,     sa,     ca]
            ])
        return np.array(R)
    elif c == 0:
        # rotation around x and y
        ca = np.cos(a)
        sa = np.sin(a)
        cb = np.cos(b)
        sb = np.sin(b)
        R = np.array([
            [cb, -sa*sb, -sb*ca],
            [ 0,     ca,    -sa],
            [sb,  sa*cb,  ca*cb]
            ])
        return np.array(R)
    else:
        # all rotations
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)
        R = np.array([
            [cb*cc, -sa*sb*cc - sb*ca,  sa*sb - sb*ca*cc],
            [sb*cb, -sa*sb*sb + ca*cc, -sa*cc - sb*sb*ca],
            [   sb,             sa*cb,             ca*cb]
            ])
        return np.array(R)
        
        
    
        
        

    
    
    
            
            
    

if __name__ == "__main__":

    v = np.zeros(3)
    u = unitaryVector(v)
        