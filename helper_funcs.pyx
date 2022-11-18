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
        return np.array(singleRotationMatrix(-b,1))
    elif a == 0.:
        # rotation around y and z
        cb = np.cos(b)
        sb = np.sin(b)
        cc = np.cos(c)
        sc = np.sin(c)
        return np.array([
            [cb*cc, -sc, sb*cc],
            [sc*cb,  cc, sb*sc],
            [-sb,    0.0, cb   ]
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
            [cb, sa*sb, sb*ca],
            [0.,     ca,    -sa],
            [-sb, sa*cb,  ca*cb]
            ])
    else:
        # all rotations
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)
        return np.array([
            [cb*cc, sa*sb*cc - sc*ca,  sa*sc + sb*ca*cc],
            [sc*cb, sa*sb*sc + ca*cc, -sa*cc + sb*sc*ca],
            [   -sb,             sa*cb,             ca*cb]
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


cpdef skew(double [:] vec):
    '''
    Returns the skew symmetric matrix associated to the vector
    
    Parameters
    ----------
    vec : array of 3 doubles
        THe input vector
    
    Returns
    -------
    skew : np.array of size 3x3
        The skew symmetric matrix associated to the vector
    '''
    return np.array([[0.0,-vec[2],vec[1]],
                     [vec[2],0,-vec[0]],
                     [-vec[1],vec[0],0.0]])


cpdef minkowskiSum(double [:,:] P, double [:,:] Q):
    '''
    Returns the Minkowski sum of two polygons P and Q
    
    The Minkowski sum is also called Configuration Space Obstacle (CSO) of P 
    and Q and is defined as
    
    CSO = {x_P - x_Q:x_P in P, x_Q in Q}
    
    Parameters
    ----------
    Two memory views that contain the points of the polygons P and Q as
    ordered (x,y) pairs
    
    Returns
    -------
    An array containing the Minkowski sum
    
    '''
    
    cdef Py_ssize_t numPtsP = P.shape[0]
    cdef Py_ssize_t numPtsQ = Q.shape[0]
    cdef Py_ssize_t i,j
    
    cdef double[:,:] cso = np.zeros((numPtsP*numPtsQ,2))
    
    print(np.array(cso))
    for i in range(numPtsP):
        for j in range(numPtsQ):
            print('{}|{}|{}'.format(i*numPtsQ+j,i,j))
            cso[i*numPtsQ+j][0] = P[i][0]-Q[j][0]
            cso[i*numPtsQ+j][1] = P[i][1]-Q[j][1]
            
    return cso

cpdef gjk(double [:,:] P, double [:,:] Q, double[:] v0):
    '''
    Implementation of the GJK algorithm as of Montanari et alii.
    
    Reference:
    MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
    for Faster and More Reliable Distance Queries Between Convex Objects. 
    ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 
    
    '''
    cdef int k = 0
    cdef double [2] wk
    cdef list tauk, Wk, lam
    cdef Py_ssize_t idP,idQ,i
    
    tauk = []
    Wk = []
    vk = np.array(v0)
    wk[0] = 0
    wk[1] = 0
    
    while len(Wk) != 4 or (vk[0]*vk[0]+vk[1]*vk[1]) > 1e-6:
        # increments the number of iterations
        k += 1
        idP = supportFunction(P,vk)
        idQ = supportFunction(Q,-vk)
        wk[0] = P[idP,0] - Q[idQ,0]
        wk[1] = P[idP,1] - Q[idQ,1]
        tauk = []
        for i in range(len(Wk)):
            tauk.append(Wk[i])
        tauk.append(wk)
        Wk,lam = signedVolumesDistance(tauk)
        
        vk = 0*vk
        for i in range(len(Wk)):
            vk += lam[i]*Wk[i]
            
        print('Iteraction number {}'.format(k))
        print('wk = {}'.format(np.array(wk)))
        print('tauk = {}'.format(tauk))
        print('Wk = {}'.format(Wk))
        print('lambda = {}'.format(lam))
        print('vk+1 = {}'.format(vk))
        if k > 20:
            break
        
    return np.norm(vk)

cpdef supportFunction(double [:,:] P, double [:] v):
    '''
    Get the support function value for a convex polygon along
    a specified direction.
    
    The support function value is
    max{k.v}, with k in P
    
    Returns
    -------
    The index of the element in P that satisfies the support function condition
    '''
    
    cdef Py_ssize_t i, maxIndex
    cdef Py_ssize_t numPoints = P.shape[0]
    cdef double currSvalue, maxValue
    
    maxIndex = 0
    maxValue = -1e32
    for i in range(numPoints):
        currSvalue = P[i][0]*v[0] + P[i][1]*v[1]
        if currSvalue > maxValue:
            maxValue = currSvalue
            maxIndex = i
            
    return maxIndex

cpdef signedVolumesDistance(list tau):
    '''
    Signed volumes distance subalgorithm to be used with the GJK method, as
    proposed by Montanari et al.
    
    Reference:
    MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
    for Faster and More Reliable Distance Queries Between Convex Objects. 
    ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 
    
    Returns
    -------
    W = the subset of vertices that contain the minimal distance point in 
    its convex hull
    lam = the weights
    '''
    cdef list W = []
    cdef list lam = []
    cdef Py_ssize_t i
    
    if len(tau) == 4:
        pass
    elif len(tau) == 3:
        pass
    elif len(tau) == 2:
        t = tau[1] - tau[0]
        po = tau[1].dot(t)/(t.dot(t)) * t + s[1]
        mumax = 0
        for i in range(2):
            mu = tau[0][i]-tau[1][i]
            if mu*mu > mumax*mumax:
                mumax = mu
    else:
        lam = [1]
        W = tau
        
    return W, lam
    
        
    
    
    
            
        
if __name__ == "__main__":

    v = np.zeros(3)
    u = unitaryVector(v)
        