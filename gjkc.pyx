#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Created on Thu Jan 19 23:25 2023

Cython version of the GJK algorithm

[1] C. Esperança, “2D GJK and EPA algorithms.” 
https://observablehq.com/@esperanc/2d-gjk-and-epa-algorithms 
(accessed Jan. 09, 2023).
[2] M. Montanari, N. Petrinic, and E. Barbieri, 
“Improving the GJK Algorithm for Faster and More Reliable 
Distance Queries Between Convex Objects,” 
ACM Trans. Graph., vol. 36, no. 3, pp. 1–17, Jun. 2017, doi: 10.1145/3083724.
[3] G. V. den Bergen, “A Fast and Robust GJK Implementation for 
Collision Detection of Convex Objects,” 
Journal of Graphics Tools, vol. 4, no. 2, pp. 7–25, 
Jan. 1999, doi: 10.1080/10867651.1999.10487502.


@author: leonardo
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from matplotlib.pyplot import plot, fill, arrow, gca,\
    sca, figure, axes, close, show


@cython.boundscheck(False)

cpdef gjk(double[:, :] P, double[:, :] Q, double[:] v0, 
          bint graphs=False, bint verb=False):
    """
    Implement the GJK algorithm as of Montanari et alii.

    Reference:
    MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
    for Faster and More Reliable Distance Queries Between Convex Objects. 
    ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 
    """
    
    ## Integers
    # this is the problem dimension (2D or 3D)
    cdef int dims = P.shape[1]
    
    cdef int k = 0
    cdef int maxiter = 20
    cdef int idP, idQ
    cdef Py_ssize_t iP, iQ, li, initPointIdx, finalPointIdx, lami
    cdef int i, j, ni, lenWk
    ## Doubles
    cdef double epst = 1e-8
    cdef double epsr = 1e-3
    cdef double contactDistance = 1.0
    cdef double vknorm2
    cdef double maxmody2, sTol, dist
    ## Double vectors
    cdef double[:] wk = np.zeros(dims)
    cdef double[:] contactNormal = np.zeros(dims)
    cdef double[:] a, b
    cdef double[:] normalToEdge
    cdef double [:] lam
    cdef double[:] vk = np.array(v0, dtype=np.float64)
    cdef double[:] neg_vk = np.array(v0, dtype=np.float64)
    cdef double[:] r = np.zeros(dims, dtype=np.float64)
    ## Double arrays
    cdef double [:,:] tauk = None 
    cdef double [:,:] Wk = None
    cdef double [:,:] simplex
    cdef double [:,:] edge    
    ## Integer arrays
    cdef int[:] tempArr1, tempArr2
    ## Others
    cdef bint iswkintk = False
    cdef list indexP = []
    cdef list idxListP, idxListQ
    cdef list indexQ = []

    # Attributions
    vknorm2 = np.dot(vk,vk)
    
    if graphs:
        figure()
        for i in range(P.shape[0]):
            for j in range(Q.shape[0]):
                plot(P[i, 0] - Q[j, 0], P[i, 1] - Q[j, 1], '*')
        show()
        ax = gca()

    ni = 0
    i = 0
    
    lenWk = 0
    while lenWk < 4 and k < maxiter:
        k += 1
        #print('Iteration {}'.format(k))
        for i in range(vk.shape[0]):
            neg_vk[i] = -vk[i]
        idP = supportFunction(P, neg_vk)
        idQ = supportFunction(Q, vk)
        wk[0] = P[idP, 0] - Q[idQ, 0]
        wk[1] = P[idP, 1] - Q[idQ, 1]
        #print('vk = {}'.format(np.array(vk)))
        #print('idP = {} | idQ = {}'.format(idP,idQ))
        #print('wk = {}'.format(np.array(wk)))

        if tauk is not None:
            iswkintk = np.any(np.all(wk == np.array(tauk), axis=1))
        if iswkintk:
            if verb:
                print(f'Algorithm terminated after {k} iterations because next vertex is already in the simplex.')
            break

        if (vknorm2 - np.dot(vk, wk)) <= epsr * epsr * vknorm2:
            if verb:
                print('Algorithm terminated because vector is close enough.')
            break
        
        tauk = np.zeros((lenWk+1,dims))
        for i in range(lenWk):
            for j in range(dims):
                tauk[i,j] = Wk[i,j]
                
        for j in range(dims):
            tauk[lenWk,j] = wk[j]
        #print('tauk = {}'.format(np.array(tauk)))
        
        indexP.append(idP)
        indexQ.append(idQ)
        Wk, lam, tempArr1, tempArr2 = signedVolumesDistance(tauk,
                                                        np.array(indexP,
                                                                 dtype=np.int32),
                                                        np.array(indexQ,
                                                                 dtype=np.int32))
        
        lenWk = Wk.shape[0]
        
        #print('Wk = {}'.format(np.array(Wk)))
        #print('-----------------------------')
        
        indexP = []
        indexQ = []
        
        for i in range(tempArr1.shape[0]):
            indexP.append(tempArr1[i])
            indexQ.append(tempArr2[i])
        

        if graphs:
            wplot = np.array(Wk)
            fill(wplot[:, 0], wplot[:, 1], facecolor='blue', edgecolor='purple', alpha=0.5)
            show()

        
        for j in range(dims):
            vk[j] = 0
        maxmody2 = 0
        for i in range(lenWk):
            for j in range(dims):
                vk[i] += lam[i] * Wk[i,j]
            if np.dot(Wk[i], Wk[i]) > maxmody2:
                maxmody2 = np.dot(Wk[i], Wk[i])

        vknorm2 = np.dot(vk, vk)
        
        if graphs:
            arrow(0.0, 0.0, vk[0], vk[1], color='red', length_includes_head=True)

        if vknorm2 <= epst * maxmody2:
            break

    contactDistance = sqrt(vknorm2)

    lami = 0
    if Wk.shape[0] == 3:
        sTol = 1e-6
        simplex = np.array(Wk)
        idxListP = indexP.copy()
        idxListQ = indexQ.copy()
        k = 0
        lam = np.zeros(200)
        while k < 100:
            initPointIdx, finalPointIdx, normalToEdge, dist = closestEdge(simplex)
            edge = np.zeros((2,dims))
            for i in range(2):
                for j in range(dims):
                    edge[i,j] = simplex[finalPointIdx,j] - simplex[initPointIdx,j]
            if graphs:
                arrow(simplex[initPointIdx][0], simplex[initPointIdx][1],
                      edge[0], edge[1], color='cyan')
            iP = supportFunction(P, -np.array(normalToEdge))
            iQ = supportFunction(Q, normalToEdge)
            for i in range(dims):
                r[i] = P[iP,i] - Q[iQ,i]
            if np.abs(np.dot(normalToEdge, r)) - dist < sTol:
                p1 = np.array(simplex[initPointIdx])
                p2 = np.array(simplex[finalPointIdx])
                detA = p1[0] * p2[1] - p1[1] * p2[0]
                lam[lami] = (-(dist * normalToEdge[0] * p2[1] -\
                             dist * normalToEdge[1] * p2[0]) / detA)
                lami += 1
                lam[lami] = (-(p1[0] * dist * normalToEdge[1] -\
                             p1[1] * dist * normalToEdge[0]) / detA)
                lami += 1
                
                indexP = [idxListP[initPointIdx], idxListP[finalPointIdx]]
                indexQ = [idxListQ[initPointIdx], idxListQ[finalPointIdx]]
                break
            simplex = np.insert(simplex, initPointIdx, r, axis=0)
            idxListP.insert(initPointIdx, iP)
            idxListQ.insert(initPointIdx, iQ)
            k += 1

        contactNormal = normalToEdge
        contactDistance = -dist

    a = np.zeros(dims, dtype=np.float64)
    b = np.zeros(dims, dtype=np.float64)
    for li in range(lami):
        for j in range(dims):
            a[j] += lam[li] * P[indexP[li],j]
            b[j] += lam[li] * Q[indexQ[li],j]

    return np.array(a), np.array(b), np.array(contactNormal), contactDistance


cpdef closestEdge(double [:,:] Wk):
    """
    Receive a simplex and returns the closest edge to the origin.

    Parameters
    ----------
    Wk : list of arrays
        a 2-simplex with three vertices

    Returns
    -------
    None.
    """
    
    cdef int npts = Wk.shape[0]
    cdef int l = npts - 1
    
    cdef int I, J, i, j   
    
    cdef double minDist = np.inf
    cdef double normd
    cdef double[:] initPoint = np.zeros(2)
    cdef double[:] edge = np.zeros(2)
    cdef double[:] normalToEdge = np.zeros(2)
    cdef double[:] e = np.zeros(2)
    cdef double[:] eunit = np.zeros(2)
    cdef double[:] d   
    
    I = 0
    J = 0

    for i in range(npts):
        norme = 0
        for j in range(2):
            e[j] = Wk[l,j] - Wk[i,j]
            norme += e[j] * e[j]
        norme = np.sqrt(norme)
        for j in range(2):
            if norme != 0:
                eunit[j] = e[j]/norme
            else:
                eunit[j] = e[j]

        # d = np.linalg.norm(Wk[i]-Wk[i].dot(eunit)*eunit)
        d = np.array([eunit[1]*eunit[0]*Wk[i,1]-eunit[1]*eunit[1]*Wk[i,0],
                      -eunit[0]*eunit[0]*Wk[i,1]+eunit[0]*eunit[1]*Wk[i,0]])
        normd = sqrt(np.dot(d,d))
        

        if normd < minDist:
            minDist = normd
            for j in range(2):
                initPoint[j] = Wk[i,j]
            edge = e
            for j in range(2):
                if normd != 0:
                    normalToEdge[j] = d[j]/normd
                else:
                    normalToEdge[j] *= 0
            I = i
            J = l

        l = i

    return I, J, normalToEdge, minDist


cpdef supportFunction(double [:,:] P, double [:] v):
    """
    Get the support function value for a convex polygon along a specified direction.

    The support function value is
    max{k.v}, with k in P

    Returns
    -------
    The index of the element in P that satisfies the support function condition
    """
    cdef int i = 0
    cdef int maxIndex = 0
    cdef int numPoints = P.shape[0]
    cdef double currSvalue = 0.
    cdef double maxValue = 0.
    
    # i, maxIndex = 0, 0
    # numPoints = P.shape[0]
    # currSvalue, maxValue = 0., 0.

    maxValue = -np.Inf
    for i in range(numPoints):
        currSvalue = P[i,0]*v[0] + P[i,1]*v[1]
        if currSvalue > maxValue:
            maxValue = currSvalue
            maxIndex = i

    return maxIndex


cpdef compareSigns(double x, double [:] y):
    """
    Compare the signs of two numbers.

    Parameters
    ----------
    x : number
        DESCRIPTION.
    y : number
        DESCRIPTION.

    Returns
    -------
    int
        1, if x > 0 and y >0,
        1, if x < 0 and y < 0,
        0, toherwise

    """
    cdef int i = 0
    cdef int n = y.shape[0]
    
    # Check if x is greater than 0 and all elements of y are greater than or equal to 0
    if x > 0:
        for i in range(n):
            if y[i] < 0:
                break
        else:
            return True

    # Check if x is less than 0 and all elements of y are less than or equal to 0
    elif x < 0:
        for i in range(n):
            if y[i] > 0:
                break
        else:
            return True

    return False

cpdef signedVolumesDistance(double [:,:] tau, int [:] indexP, int [:] indexQ):
    """
    Find signed volume distance.

    Signed volumes distance subalgorithm to be used with the GJK method, as
    proposed by Montanari et al.

    Reference:
    MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
    for Faster and More Reliable Distance Queries Between Convex Objects. 
    ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 


    Paramters:
    ---------
    tau - the simplex that should be analyzed for signed volumes distance
    indexP - the indices of the elements in P that are represented in tau
    indexQ - the indices of the elements in Q that are represented in tau

    Returns
    -------
    W = the subset of vertices that contain the minimal distance point in 
    its convex hull
    lam = the weights
    """
    
    cdef double[:,:] W, Wstar,s
    cdef double[:] lam, la, lamstar, dstar, t, po
    cdef int [:] idxP, idxQ
    cdef int i, j, k, m, b, bb
    cdef double [:] C = np.zeros(tau.shape[0], dtype=np.float64)
    cdef double mumax, dstar2
    cdef double tempC, tsquared

    cdef int dims = tau.shape[1]

    if tau.shape[0] == 4:
        print('Error')
        pass
    elif tau.shape[0] == 3:
        #######################################
        # subroutine SD2: search on 2-simplex
        #######################################
        # because we are dealing only wit a plane problem, vectors
        # n and p pf the original algorithm can be disregarded,
        # because p0 = 0 for the 2D case
        mumax = 0
        k = 0

        # the next part is unnecessary for the 2d case, because it computes
        # the projection of maximum area of the simplex of the three cartesian
        # planes. Since we are working on a single plane, the project is the
        # simplex itself.
        # UNCOMMENT AND MODIFY FOR 3D CASE:
        # for i in range(2):
        #     mu = tau[1][k]*tau[2][l] + tau[0][k]*tau[1][l] + tau[2][k]*tau[0][l] - tau[1][k]*tau[0][l] - tau[2][k]*tau[1][l] - tau[0][k]*tau[2][l]
        #     if mu*mu > mumax*mumax:
        #         mumax = mu
        #     k = l
        #     l = i

        # twice the area of the simples
        mumax = tau[0,0]*tau[1,1] + tau[1,0]*tau[2,1] + tau[0,1]*tau[2,0] \
            - tau[1,1]*tau[2,0] - tau[2,1]*tau[0][0] - tau[1,0]*tau[0,1]

        # get the solution matrix minors
        C[0] = 1 * (tau[1,0]*tau[2,1] - tau[2,0]*tau[1,1])
        C[1] = -1 * (tau[0,0]*tau[2,1] - tau[2,0]*tau[0,1])
        C[2] = 1 * (tau[0,0]*tau[1,1] - tau[1,0]*tau[0,1])

        if compareSigns(mumax, C):
            # if all minors determinants have the same sign as mumax,
            # then the origin is inside the simplex and the polygons intersect
            lam = np.zeros(3)
            la = np.zeros(3)
            W = np.zeros((3,dims))
            for i in range(3):
                # the following lines are from the original Montanari's algorithm
                # they return the original simplex, i.e., a simplex
                # that contains the origin
                la[i] = C[i]/mumax
                W[i] = tau[i]
                # do these belong to this for loop? I think not, but TODO
                idxP = indexP
                idxQ = indexQ
        else:
            # else the closest side of the simplex is found and selected as
            # the new simplex
            for j in range(3):
                d = 1e16
                s = np.zeros((2,dims))
                iP = np.zeros(2, dtype=np.int32)
                iQ = np.zeros(2, dtype=np.int32)
                for m in range(3):
                    b = 0
                    if m != j:
                        for bb in range(dims):
                            s[b,bb] = tau[m,bb]
                        iP[b] = indexP[m]
                        iQ[b] = indexQ[m]
                        b += 1
                if compareSigns(mumax, np.array([-C[j]])):
                    Wstar, lamstar, iP, iQ = signedVolumesDistance(s, iP, iQ)
                    dstar = np.zeros(dims)
                    for i in range(dims):
                        for j in range(Wstar.shape[1]):
                            dstar[i] += lamstar[i]*Wstar[i,j]
                    dstar2 = 0
                    for i in range(dims):
                        dstar2 += dstar[i]*dstar[i]
                    if dstar2 < d*d:
                        W = Wstar
                        lam = lamstar
                        idxP = iP
                        idxQ = iQ
                        d = np.sqrt(dstar2)

        # END OF SD2 ############################
    elif tau.shape[0] == 2:
        #######################################
        # subroutine SD1: search on 1-simplex
        #######################################
        lam = np.zeros(2)
        t = np.zeros(tau.shape[1])
        tsquared = 0.0
        po = np.zeros(tau.shape[1])
        for i in range(tau.shape[1]):
            t[i] = tau[1,i] - tau[0,i]
            tsquared += t[i] * t[i]
        # next line is wrong on Montanari's paper
        # we have to subtract tau[1] from the projection
        for i in range(tau.shape[1]):
            po[i] = tau[1,i] - tau[1,i]*t[i]/(tsquared) * t[i]
        mumax = 0
        for i in range(2):
            mu = tau[0,i]-tau[1,i]
            if mu*mu > mumax*mumax:
                mumax = mu
                I = i
        k = 1
        for j in range(2):
            C[j] = (-1)**(j+1)*(tau[k,I]-po[I])
            k = j
        if compareSigns(mumax, C):
            for i in range(2):
                lam[i] = C[i]/mumax
            W = tau
            idxP = indexP
            idxQ = indexQ
        else:
            lam = np.array([1.])
            W = np.zeros((1,dims))
            for j in range(dims):
                W[0,i] = tau[1,j]
            idxP = np.array([indexP[1]], dtype = np.int32)
            idxQ = np.array([indexQ[1]], dtype = np.int32)

        # END OF SD1 ############################
    else:
        lam = np.array([1.])
        W = tau
        idxP = indexP
        idxQ = indexQ

    return W, lam, idxP, idxQ