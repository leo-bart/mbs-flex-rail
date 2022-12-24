#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:57:38 2022

@author: leonardo
"""
import numpy as np
import helper_funcs as hf
from matplotlib.pyplot import plot, fill, arrow, gca

def gjk(P, Q, v0):
    '''
    Implementation of the GJK algorithm as of Montanari et alii.
    
    Reference:
    MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
    for Faster and More Reliable Distance Queries Between Convex Objects. 
    ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 
    
    '''
    k = 0
    wk = np.zeros(2,dtype=float)
    tauk = []
    Wk = []
    lam = []
    idP,idQ,i = 0,0,0
    idPout, idQout = 0,0
    
    tauk = []
    Wk = []
    vk = np.array(v0)
    
    goon = True
    iswkinWk = False
    indexP = []
    indexQ = []
    
    while len(Wk) < 4 and goon and k < 20:
        # increments the number of iterations
        k += 1
        idP = supportFunction(P,-vk)
        idQ = supportFunction(Q,vk)
        wk[0] = P[idP,0] - Q[idQ,0]
        wk[1] = P[idP,1] - Q[idQ,1]
        
        # first exit condition: wk is in Wk
        if len(Wk):
            iswkinWk = np.any(np.all(wk==Wk,axis=1))
        if iswkinWk:
            break
        
        # second exit condition: vk is close enough
        if (vk.dot(vk)*(1-1e-16) - vk.dot(wk)) <= 0:
            break
        
        tauk = []
        for i in range(len(Wk)):
            tauk.append(Wk[i])
        # a copy is required, otherwise the original vector is changed every iteration
        tauk.append(wk.copy())
        indexP.append(idP)
        indexQ.append(idQ)
        Wk,lam,indexP,indexQ = signedVolumesDistance(tauk,indexP,indexQ)
        
        vk = 0*vk
        maxmody = 0
        for i in range(len(Wk)):
            vk += lam[i]*Wk[i]
            if np.linalg.norm(Wk[i]) > maxmody:
                maxmody = np.linalg.norm(Wk[i])
        
        goon = vk.dot(vk) > maxmody*maxmody * 1e-8
        
    
    a = np.array([0,0],dtype=float)
    for li in range(len(lam)):
        a += lam[li] * P[indexP[li]]
    b = a - vk
    
        
    return a, b

def supportFunction(P,v):
    '''
    Get the support function value for a convex polygon along
    a specified direction.
    
    The support function value is
    max{k.v}, with k in P
    
    Returns
    -------
    The index of the element in P that satisfies the support function condition
    '''
    
    i, maxIndex = 0,0
    numPoints = P.shape[0]
    currSvalue, maxValue = 0.,0.
    
    maxIndex = 0
    maxValue = -1e32
    for i in range(numPoints):
        currSvalue = P[i][0]*v[0] + P[i][1]*v[1]
        if currSvalue > maxValue:
            maxValue = currSvalue
            maxIndex = i
            
    return maxIndex

def signedVolumesDistance(tau,indexP,indexQ):
    '''
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
    '''
    W = []
    lam = []
    i = 0
    C = np.zeros(len(tau),dtype=float)
    
    def compareSigns(x,y):
        '''
        Compare the signs of two numbers

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

        '''
        if x > 0 and (y>=0).all():
            return True
        elif x < 0 and (y<=0).all():
            return True
        else:
            return False
    
    if len(tau) == 4:
        pass
    elif len(tau) == 3:
        #######################################
        # subroutine SD2: search on 2-simplex
        #######################################
        # because we are dealing only wit a plane problem, vectors
        # n and p pf the original algorithm can be disregarded,
        # because p0 = 0 for the 2D case
        mumax = 0
        k = 0 
        l = 1
        
        d = 1e16
        
        for i in range(2):
            mu = tau[1][k]*tau[2][l] + tau[0][k]*tau[1][l] + tau[2][k]*tau[0][l] - tau[1][k]*tau[0][l] - tau[2][k]*tau[1][l] - tau[0][k]*tau[2][l]
            if mu*mu > mumax*mumax:
                mumax = mu
            k = l
            l = i
        
        k = 1
        l = 2
        for j in range(3):
            C[j] = (-1)**(j+1)*(tau[k][0]*tau[l][1] - tau[k][1]*tau[l][0])
            if C[j] == 0:
                C[j] = 0.0
            k = l
            l = i
        if compareSigns(mumax,-C):
            for i in range(3):
                lam.append(C[i]/mumax)
                W.append(tau[i])
                idxP = indexP
                idxQ = indexQ
        else:
            for j in range(3):
                s = []
                iP = []
                iQ = []
                for m in range(3):
                    if m != j:
                        s.append(tau[m])
                        iP.append(indexP[m])
                        iQ.append(indexQ[m])
                if compareSigns(mumax,C[j]):
                    Wstar,lamstar, iP, iQ = signedVolumesDistance(s,iP,iQ)
                    dstar = 0
                    for i in range(len(Wstar)):
                        dstar += lamstar[i]*Wstar[i]
                    if dstar.dot(dstar) < d*d:
                        W = Wstar
                        lam = lamstar
                        idxP = iP
                        idxQ = iQ
                        d = np.sqrt(dstar.dot(dstar))
                        
                        
        # END OF SD2 ############################
    elif len(tau) == 2:
        #######################################
        # subroutine SD1: search on 1-simplex
        #######################################
        t = tau[1] - tau[0]
        # next line is wrong on Montanari's paper
        # we have to subtract tau[1] from the projection
        po = tau[1] - tau[1].dot(t)/(t.dot(t)) * t
        mumax = 0
        for i in range(2):
            mu = tau[0][i]-tau[1][i]
            if mu*mu > mumax*mumax:
                mumax = mu
                I = i
        k = 1
        for j in range(2):
            C[j] = (-1)**(j+1)*(tau[k][I]-po[I])
            k = j
        if compareSigns(mumax, C):
            for i in range(2):
                lam.append(C[i]/mumax)
            W = tau
            idxP = indexP
            idxQ = indexQ
        else:
            lam = [1]
            W = [tau[0]]
            idxP = [indexP[0]]
            idxQ = [indexQ[0]]
        
        # END OF SD1 ############################
    else:
        lam = [1]
        W = tau
        idxP = indexP
        idxQ = indexQ
        
    return W, lam, idxP, idxQ



if __name__ == '__main__':
    P = np.array([[0.6,2],[1.6,1],[2.6,2],[1.6,3]],dtype=float)
    Q = np.array([[2,1.5],[4.,0.5],[2,-0.5]],dtype=float)
    v = np.array([0,-1],dtype=float)
    
    fill(P[:,0],P[:,1])
    fill(Q[:,0],Q[:,1])
    
    a,b = gjk(P,Q,v)
    
    arrow(b[0],b[1],a[0]-b[0],a[1]-b[1],color='red')
    
    