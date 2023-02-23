#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:57:38 2022

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
from matplotlib.pyplot import plot, fill, arrow, gca, sca, figure, axes, close, show

def gjk(P, Q, v0, graphs=False, verb=False):
    """
    

    Parameters
    ----------
    P : TYPE
        DESCRIPTION.
    Q : TYPE
        DESCRIPTION.
    v0 : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.


    Implement the GJK algorithm as of Montanari et alii.
    
    Reference:
    MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
    for Faster and More Reliable Distance Queries Between Convex Objects. 
    ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 
    
    """
    epst = 1e-8
    epsr = 1e-3
    
    k = 0
    wk = np.zeros(2,dtype=float)
    tauk = []
    Wk = []
    lam = []
    idP,idQ,i = 0,0,0
    
    tauk = []
    Wk = []
    vk = np.array(v0)
    vknorm2 = vk.dot(vk)
    
    iswkintk = False
    indexP = []
    indexQ = []
    
    contactNormal = np.array([0,0], dtype=float)
    contactDistance = 1.0
    
    if graphs:
        figure()
        for i in P:
            for j in Q:
                plot(i[0]-j[0],i[1]-j[1],'*')
        show()
        ax = gca()
    
    while len(Wk) < 4 and k < 20:
        # increments the number of iterations
        k += 1
        idP = supportFunction(P,-vk)
        idQ = supportFunction(Q,vk)
        wk[0] = P[idP,0] - Q[idQ,0]
        wk[1] = P[idP,1] - Q[idQ,1]
        
        # first exit condition: wk is in Wk
        if len(tauk):
            iswkintk = np.any(np.all(wk==tauk,axis=1))
        if iswkintk:
            if verb:
                print('Algorithm terminated after {} iterations because next vertex is already in the simplex.'.format(k))
            break
        
        # second exit condition: ~vk is close enough
        # if (vknorm2*(1-1e-16) - vk.dot(wk)) <= 0:
        if (vknorm2-vk.dot(wk)) <= epsr*epsr*vknorm2:
            if verb:
                print('Algorithm terminated because vector is close enough.')
            break
        
        tauk = Wk
        # a copy is required, otherwise the original vector is changed every iteration
        tauk.append(wk.copy())
        indexP.append(idP)
        indexQ.append(idQ)
        Wk,lam,indexP,indexQ = signedVolumesDistance(tauk,indexP,indexQ)

        if graphs:
            wplot = np.array(Wk)
            fill(wplot[:,0],wplot[:,1],facecolor='blue',edgecolor='purple',alpha=0.5)
            show()
        
        vk = 0*vk
        maxmody2 = 0
        for i in range(len(Wk)):
            vk += lam[i]*Wk[i]
            if Wk[i].dot(Wk[i]) > maxmody2:
                maxmody2 = Wk[i].dot(Wk[i])
                
        vknorm2 = vk.dot(vk)
        
        if graphs: 
            arrow(0.0,0.0,vk[0],vk[1],color='red', length_includes_head=True)
        
        if vknorm2 <= epst * maxmody2:
            break
        
        
    contactDistance = np.sqrt(vknorm2)
         
    # now we've got to treat the contact
    if len(Wk) == 3:
        #print('Contact')
        
        # next lines verify the closest edge subalgorithm      
        # initPointIdx, finalPointIdx, normal, dist = closestEdge(Wk)
        # edge = Wk[finalPointIdx] - Wk[initPointIdx]
        # arrow(Wk[initPointIdx][0],Wk[initPointIdx][1],edge[0],edge[1],color='cyan')
        
        
        # now we use the expanding polytope algorithm (EPA) to track down the
        # penetration depth
        sTol = 1e-6                         # search tolerance
        simplex = Wk.copy()
        idxListP = indexP.copy()
        idxListQ = indexQ.copy()
        k = 0
        while k < 100:
            # closest edge to the origin
            initPointIdx, finalPointIdx, n, dist = closestEdge(simplex)
            edge = simplex[finalPointIdx] - simplex[initPointIdx]
            if graphs:
                arrow(simplex[initPointIdx][0],
                      simplex[initPointIdx][1],
                      edge[0],
                      edge[1],
                      color='cyan')
            iP = supportFunction(P, -n)
            iQ = supportFunction(Q, n)
            r = P[iP] - Q[iQ]
            if (np.abs(n.dot(r))-dist < sTol):
                lam = []
                p1=simplex[initPointIdx]
                p2=simplex[finalPointIdx]
                detA = p1[0]*p2[1] - p1[1]*p2[0]
                lam.append(-(dist*n[0]*p2[1] - dist*n[1]*p2[0])/detA)
                lam.append(-(p1[0]*dist*n[1] - p1[1]*dist*n[0])/detA)
                indexP = [idxListP[initPointIdx],idxListP[finalPointIdx]]
                indexQ = [idxListQ[initPointIdx],idxListQ[finalPointIdx]]
                break
            simplex.insert(initPointIdx+1,r)
            idxListP.insert(initPointIdx+1,iP)
            idxListQ.insert(initPointIdx+1,iQ)
            k += 1
            
        contactNormal = n
        contactDistance = -dist
            
        
        
    
    a = np.array([0,0],dtype=float)
    b = np.array([0,0],dtype=float)
    for li in range(len(lam)):
        a += lam[li] * P[indexP[li]]
        b += lam[li] * Q[indexQ[li]]
        
    #b = a - vk
        
        
    return a, b, contactNormal, contactDistance

def closestEdge(Wk):
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
    npts = len(Wk)
    l = npts - 1
    minDist = np.inf
    initPoint = np.array([0.,0.])
    edge = np.array([0.,0.])
    I = 0
    J = 0
    for i in range(npts):
        e = Wk[l] - Wk[i]
        norme = np.sqrt(e.dot(e))
        if norme != 0:
            eunit = e/norme
        else:
            eunit = e
             
        d = np.linalg.norm(Wk[i]-Wk[i].dot(eunit)*eunit)
        
        if d < minDist:
            minDist = d
            initPoint = Wk[i]
            edge = e
            I = i
            J = l
        
        l = i
        
    normalToEdge = np.array([edge[1]*edge[0]*initPoint[1]-edge[1]*edge[1]*initPoint[0],
                            -edge[0]*edge[0]*initPoint[1]+edge[0]*edge[1]*initPoint[0]])
    normalToEdge = normalToEdge/np.linalg.norm(normalToEdge)
        
    return I, J, normalToEdge, minDist
    

def supportFunction(P,v):
    """
    Get the support function value for a convex polygon along a specified direction.
    
    The support function value is
    max{k.v}, with k in P
    
    Returns
    -------
    The index of the element in P that satisfies the support function condition
    """
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
    W = []
    lam = []
    i = 0
    C = np.zeros(len(tau),dtype=float)
    
    def compareSigns(x,y):
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
        if x > 0 and (y>=0).all():
            return True
        elif x < 0 and (y<=0).all():
            return True
        else:
            return False
    
    if len(tau) == 4:
        print('Error')
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
        
        # the next part is unnecessary for the 2d case, because it computes
        # the projection of maximum area of the simplex of the three cartesian
        # planes. Since we are working on a single plane, the project is the
        # simplex itself.
        ### UNCOMMENT AND MODIFY FOR 3D CASE:
        # for i in range(2):
        #     mu = tau[1][k]*tau[2][l] + tau[0][k]*tau[1][l] + tau[2][k]*tau[0][l] - tau[1][k]*tau[0][l] - tau[2][k]*tau[1][l] - tau[0][k]*tau[2][l]
        #     if mu*mu > mumax*mumax:
        #         mumax = mu
        #     k = l
        #     l = i
        
        # twice the area of the simples
        mumax = tau[0][0]*tau[1][1] + tau[1][0]*tau[2][1] + tau[0][1]*tau[2][0] \
            - tau[1][1]*tau[2][0] - tau[2][1]*tau[0][0] - tau[1][0]*tau[0][1]
        
        
        # get the solution matrix minors
        C[0] = 1 * (tau[1][0]*tau[2][1] - tau[2][0]*tau[1][1])
        C[1] = -1 * (tau[0][0]*tau[2][1] - tau[2][0]*tau[0][1])
        C[2] = 1 * (tau[0][0]*tau[1][1] - tau[1][0]*tau[0][1])
        
        
        if compareSigns(mumax,C):
            # if all minors determinants have the same sign as mumax,
            # then the origin is inside the simplex and the polygons intersect
            for i in range(3):
                # the following lines are from the original Montanari's algorithm
                # they return the original simplex, i.e., a simplex
                # that contains the origin
                lam.append(C[i]/mumax)
                W.append(tau[i])
                idxP = indexP
                idxQ = indexQ
        else:
            # else the closest side of the simplex is found and selected as
            # the new simplex
            for j in range(3):
                d = 1e16
                s = []
                iP = []
                iQ = []
                for m in range(3):
                    if m != j:
                        s.append(tau[m])
                        iP.append(indexP[m])
                        iQ.append(indexQ[m])
                if compareSigns(mumax,-C[j]):
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
            W = [tau[1]]
            idxP = [indexP[1]]
            idxQ = [indexQ[1]]
        
        # END OF SD1 ############################
    else:
        lam = [1]
        W = tau
        idxP = indexP
        idxQ = indexQ
        
    return W, lam, idxP, idxQ



if __name__ == '__main__':
    close('all') 
    
    
    # P = np.array([[0.,2],[1.,1],[2.,2],[1.,3]],dtype=float)
    # Q = np.array([[1,0],[2.,1.75],[3,0]],dtype=float)
    P = np.array([[0.,1],[1.,1],[1.1,0],[0.,0]],dtype=float)
    Q = np.array([[1,0.5],[2.,1.],[2.,0]],dtype=float)
    v = np.array([0,-1],dtype=float)
    
    P = np.array([[0.49986885, 0.07875599],
           [0.50019885, 0.08149599],
           [0.50113885, 0.08430599],
           [0.50249885, 0.08666599],
           [0.50400885, 0.08846599],
           [0.50727885, 0.09092599],
           [0.50995885, 0.09205599],
           [0.51203885, 0.09253599],
           [0.51368885, 0.09268599],
           [0.51515885, 0.09272599],
           [0.51706885, 0.09277599],
           [0.51879885, 0.09280599],
           [0.52065885, 0.09282599]])
    Q = np.array([[0.459   , 0.0971  ],
       [0.460412, 0.091706],
       [0.463611, 0.083338],
       [0.465131, 0.080256],
       [0.46772 , 0.076653],
       [0.47348 , 0.072794],
       [0.478756, 0.071709],
       [0.479832, 0.071709],
       [0.48564 , 0.073012],
       [0.49051 , 0.07629 ],
       [0.493925, 0.081225],
       [0.495424, 0.083975]])
    
    Q = np.array([[0.459   , 0.0971  ],
                  [0.460412, 0.091706]])
    
    
    #shift
    d = np.array([0.0,0.008])
    Q += d
    
    f1 = figure()
    fill(P[:,0],P[:,1], linewidth=1, edgecolor='black')
    fill(Q[:,0],Q[:,1], linewidth=1, edgecolor='black')
    
    a,b,n,d  = gjk(P,Q,v)
    g = b-a
    
    if d <= 0.0:
        print('Contact occurred with penetration depth {}'.format(d))
    
    figure(f1.number)
    arrow(a[0],a[1],g[0],g[1],color='red',length_includes_head=True)
    
    
    ax = gca()
    ax.set_aspect('equal', 'box')