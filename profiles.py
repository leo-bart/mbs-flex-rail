#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:35:57 2022
Sets profiles to be applied to beams and rails
@author: leonardo
"""

import numpy as np
import matplotlib.pyplot as plt
import helper_funcs as hf
import MultibodySystem as MBS
from bodiesc import rigidBody

class profile(object):
    
    def __init__(self,name):
        self.name = name
        self.body = None
        self.refPointCoordinates = np.zeros(2)
        print('Profile {} created.'.format(name))
        
    def setParent(self,body):
        self.parent = body
        
class planarProfile(profile):
    
    
    def __init__(self,name, fileName='', convPar = 1):
        super().__init__(name)
        self.convPar = convPar
        
    def setProfilePointsFromFile(self, fileName, colDelimiter=';'):
        self.setProfilePoints(np.genfromtxt(fileName, delimiter=colDelimiter))

    
    def setProfilePoints(self,pArray):
        self.points = pArray
        self.convexSubsets = []
        
        pShape = self.points.shape
        
        if pShape[1] > pShape[0]:
            self.points.tranpose()
        
        self.nPoints = self.points.shape[0]
        
    
    def centerProfile(self):
        shift = np.zeros(2)
        shift[0] = 0.5 * (self.points[:,0].max() + self.points[:,0].min())
        shift[1] = self.points[:,1].max()
        
        self.points -= shift
        
    def offsetPoints(self, offset):
        self.points += offset
        
    def rotatePoints(self, angleInRad):
        R = np.array([[np.cos(angleInRad),np.sin(angleInRad)],
                      [-np.sin(angleInRad),np.cos(angleInRad)]])
        self.points = self.points.dot(R)
        
    def plotMe(self, ax = plt.gca()):
        self.createConvexSubsets()
        for cs in self.convexSubsets:
            ax.plot(cs[:,0], cs[:,1])
        
    def createConvexSubsets(self):
        
        rot = self.convPar
        currSet = []
        currSet.append(self.points[0])
        for i in range(1,self.nPoints-1):
            firstPoint = currSet[0]
            currPoint = self.points[i]
            prevPoint = self.points[i-1]
            nextPoint = self.points[i+1]
            npm1np = currPoint - prevPoint
            npnpp1 = nextPoint - currPoint
            n0np = currPoint - firstPoint
            angSign = np.sign(np.cross(npm1np,npnpp1))
            currSet.append(self.points[i])
            if angSign * rot < 0:
                self.convexSubsets.append(np.array(currSet))
                currSet = []
                currSet.append(self.points[i])
                
        self.convexSubsets.append(np.array(currSet))
        

class wheelRailContact(MBS.force):
    
    
    def __init__(self, name_='Wheel-rail contact'):
        super().__init__(name_)
        self.type = 'Wheel-rail contact'
        
    def setWheel(self,wheelProf):
        self.wheel = wheelProf
        self.wheelProfLen = wheelProf.points.shape[0]
        
    def setRail(self,railProf):
        self.rail = railProf
        self.railProfLen = railProf.points.shape[0]
        
    def searchContactPoint(self, tol = 1e-6):
        
        initialPoint = self.wheel.points[0]
        if self.rail.points[0,0] > initialPoint[0]:
            initialPoint = self.rail.points[0]
        
        finalPoint = self.wheel.points[-1]
        if self.rail.points[-1,0] < finalPoint[0]:
            finalPoint = self.rail.points[-1]
            
        
        cso = self.computeMikowskiSum()
        mini, minj = 0,0
        dist = 1000.
        for i in range(self.wheelProfLen):
            for j in range(self.railProfLen):
                thisDist = cso[i,j].dot(cso[i,j])
                if thisDist < dist:
                    dist = thisDist
                    mini, minj = i,j
        
        approxNormal = hf.unitaryVector(self.rail.points[minj+1] - self.rail.points[minj-1])[0][0:2]
        approxNormal = np.array([[0,-1],[1,0]]).dot(approxNormal)
        
        g = cso[mini,minj].dot(approxNormal)
        print(g)
                    
        # GJK algorithm
        # wi,ri = [0],[0]
        # v = cso[wi[0],ri[0]]
        # k = 0
        # W = [v]
        # close = False
        # terminate = False
        # while len(W) < 3 and not close and not terminate:
        #     print(np.sqrt(v.dot(v)))
        #     k = k+1
        #     w, wj, rj = self.getSupportFunction(-v, cso)
        #     ri.append(wj)
        #     wi.append(rj)
        #     W.append(w)
        #     newW,l = self.S1D(*W) if len(W) == 2 else self.S2D(*W)
        #     v = np.zeros(2)
        #     retainedIdx = [False] * len(W)
        #     for i in range(len(W)):
        #         try:
        #             retainedIdx[i] = (W[i] - newW[i]).all() == 0
        #             v += l[i]*newW[i]
        #         except:
        #             pass
                
        #     W = newW
        #     ri = [ri[i] for i in range(len(ri)) if retainedIdx[i]]
        #     wi = [wi[i] for i in range(len(wi)) if retainedIdx[i]]
            
            
        #     close = v.dot(v) <= 1e-6*np.max([yi.dot(yi) for yi in W])
        #     terminate = v.dot(v) - v.dot(w) > 1e-12*v.dot(v)
            
        # pi = np.zeros(2)
        # qi = np.zeros(2)
        # for i in range(len(l)):
        #     pi = l[i]*self.wheel.points[wi[i]]
        #     qi = l[i]*self.rail.points[ri[i]]
        
        return self.wheel.points[mini], self.rail.points[minj], mini, minj, approxNormal, g
    
    def getSupportFunction(self,v,cso):
        
        supWheel = np.argmax(self.wheel.points.dot(v))
        supRail = np.argmax(self.rail.points.dot(-v))
        
        return self.wheel.points[supWheel] - self.rail.points[supRail], supWheel, supRail
            
    
    def computeMikowskiSum(self):
        minkS = np.zeros((self.wheelProfLen,self.railProfLen,2))
        for i,wp in enumerate(self.wheel.points):
            for j,rp in enumerate(self.rail.points):
                minkS[i,j] = wp-rp
                
        return minkS
    
    
    def S2D(self,s1,s2,s3):
        '''
        Algorithm to compute the minimum distance of a 2-simplex (triangle)
        to the origin.
        
        This implemented version is a modification to a purely 2D case.
        
        Reference:
        MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
        for Faster and More Reliable Distance Queries Between Convex Objects. 
        ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 

        Parameters
        ----------
        s1 : array
            vertex 1.
        s2 : array
            vertex 2.
        s3 : array
            vertex 3.

        Returns
        -------
        W : list
            Contains the support vertices of the 1-simplex that supports the 
            point of minimal distance.
        lambda : list
            Coefficients of the 
        '''

        # the following is the main modifiction from Montanari's algorithm:
        # because we are dealing with a plane problem, the projection of the origin
        # on the plane that contains the simplex is always the origin itself
        p0 = np.zeros(2)
        s = [s1,s2,s3]
        mu_max = 0
        k = 0
        l = 1
        
        for i in range(2):
            mu = s2[k]*s3[l] + s1[k]*s2[l] + s3[k]*s1[l] - s2[k]*s1[l] - s3[k]*s2[l] - s1[k]*s3[l]
            if mu*mu > mu_max*mu_max:
                mu_max = mu
            k = l
            l = i
            
        k = 0
        l = 1
        
        C = np.zeros(3)
        for j in range(3):
            C[j] = (-1)**(j+1) * (p0[0]*s[k][1] + p0[1]*s[l][0] + s[k][0]*s[l][1] - p0[0]*s[l][1] - p0[1]*s[k][0] - s[l][0]*s[k][1])
            k = l
            l = j
        
        if all( self.compareSigns(mu_max,c) for c in C):
            lam = C/mu_max
            W = s
            retainedIdx = [0,1,2]
        else:
            d = 1e6
            for j in [0,1,2]:
                if self.compareSigns(mu_max, -C[j]):
                    Wstar, lamstar = self.S1D(*[s[m] for m in range(3) if m != j])
                    if len(Wstar) == 1:
                        dstar = Wstar[0]*lamstar[0]
                    else:
                        dstar = Wstar[0]*lamstar[0] + Wstar[1]*lamstar[1]
                    dstar = np.linalg.norm(dstar)
                    if dstar < d:
                        W = Wstar
                        lam = lamstar
                        d = dstar

        return W, lam
            
            
            
    def S1D(self,s1,s2):
        '''
        Computes the point with minimal norm of the simplex defined by vertices
        s1 and s2.

        Reference:
        MONTANARI, M.; PETRINIC, N.; BARBIERI, E. Improving the GJK Algorithm 
        for Faster and More Reliable Distance Queries Between Convex 
        Objects. ACM Transactions on Graphics, v. 36, n. 3, p. 1–17, 30 jun. 2017. 


        Parameters
        ----------
        s1 : array
            first simplex point.
        s2 : array
            second simplex point.

        Returns
        -------
        W : list of points
            smallest subset of the input simplex which components are needed to support
            the vector of minimal distance.
        lam : list of weights
            weights of the input vectors to obtain the point of minimal distance.

        '''
        
        t = hf.unitaryVector(s2 - s1)[0][:2]
        p0 = s2 - s2.dot(t) * t
        mu_max = 0
        for i in range(2):
            mu = s1[i] - s2[i]
            if mu*mu > mu_max*mu_max:
                mu_max = mu
                I = i
        C = np.zeros(2)
        s = [s1,s2]
        k = 1

        for j in range(2):
            C[j] = (-1)**(j+1) * (s[k][I] - p0[I])
            k = j
        if all( self.compareSigns(mu_max,c) for c in C):
            lam = C / mu_max
            W = [s1,s2]
        else:
            lam = [1]
            W = [s1]
        
        
        return W, lam
    
    
    
    def compareSigns(self,a,b):
        if a > 0 and b > 0:
            return 1
        elif a < 0 and b < 0:
            return 1
        else:
            return 0
        
        
    def evaluateForceFunction(self,*args):
        p = args[1]
        v = args[2]
        f = np.zeros_like(p)
        
        pw,pr,idxw,idxr,n, gap = self.searchContactPoint(self)
        
        if gap < 0:
            cForce = gap * 167e6 * n
        
            f[self.body1.globalDof] = -cForce
            f[self.body2.globalDof] = cForce
        
        return f
        
        
        

if __name__=='__main__':
    b = rigidBody('Trilho')
    
    w = planarProfile('wheel')
    r = planarProfile('rail', convPar=-1)
    
    b.addProfile(r)
    
    w.setProfilePointsFromFile('/home/leonardo/git/mbsim-wagons/roda.dat')
    r.setProfilePointsFromFile('/home/leonardo/git/mbsim-wagons/tr68.dat')
    
    r.centerProfile()
    r.rotatePoints(np.arctan(1/40))
    w.offsetPoints(np.array([-0.138,0.48728]))
    
    ax = plt.gca()
    #w.plotMe(ax)
    #r.plotMe(ax)
    ax.axis('equal')
    
    wr = wheelRailContact()
    wr.setWheel(w)
    wr.setRail(r)
    
    pw,pr,iw,ir,n,g = wr.searchContactPoint()
    
    #ax.plot(*pw,'o')
    #ax.plot(*pr,'o')
    ax.quiver(pr[0],pr[1],*n)
    
    w.createConvexSubsets()