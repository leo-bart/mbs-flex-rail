#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:57:38 2022

@author: leonardo
"""
import numpy as np
import helper_funcs as hf
from matplotlib.pyplot import plot, fill

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
    
    tauk = []
    Wk = []
    vk = np.array(v0)
    
    while len(Wk) < 4 or (vk[0]*vk[0]+vk[1]*vk[1]) > 1e-6:
        # increments the number of iterations
        k += 1
        idP = supportFunction(P,-vk)
        idQ = supportFunction(Q,vk)
        wk[0] = P[idP,0] - Q[idQ,0]
        wk[1] = P[idP,1] - Q[idQ,1]
        tauk = []
        for i in range(len(Wk)):
            tauk.append(Wk[i])
        # a copy is required, otherwise the original vector is changed every iteration
        tauk.append(wk.copy())
        Wk,lam = signedVolumesDistance(tauk)
        
        vk = 0*vk
        for i in range(len(Wk)):
            vk += lam[i]*Wk[i]
            
        if k > 20:
            break
        
    return np.linalg.norm(vk)

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

def signedVolumesDistance(tau):
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
    W = []
    lam = []
    i = 0
    C = np.array([0,0],dtype=float)
    
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
        if x > 0 and y.all() > 0:
            return True
        elif x < 0 and y.all() < 0:
            return True
        else:
            return 0
    
    if len(tau) == 4:
        pass
    elif len(tau) == 3:
        #######################################
        # subroutine SD2: search on 2-simplex
        #######################################
        n = np.cross(tau[1]-tau[0],tau[2]-tau[0])
        # END OF SD2 ############################
        pass
    elif len(tau) == 2:
        #######################################
        # subroutine SD1: search on 1-simplex
        #######################################
        t = tau[1] - tau[0]
        # next line is wrong on Montanari's paper
        # we have to subtract tau[1] from the projection
        po = tau[1].dot(t)/(t.dot(t)) * t - tau[1]
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
        else:
            lam = [1]
            W = [tau[0]]
        
        # END OF SD1 ############################
    else:
        lam = [1]
        W = tau
        
    return W, lam



if __name__ == '__main__':
    P = np.array([[0,2],[1,1],[2,2],[1,3]],dtype=float)
    Q = np.array([[2,1.5],[4.,0.5],[2,-0.5]],dtype=float)
    v = np.array([0,-1],dtype=float)
    
    iP = hf.supportFunction(P,-v)
    iQ = hf.supportFunction(Q,v)
    
    # fill(P[:,0],P[:,1])
    # fill(Q[:,0],Q[:,1])
    # plot(P[iP,0],P[iP,1],'*')
    # plot(Q[iQ,0],Q[iQ,1],'*')
    
    gjk(P,Q,v)

