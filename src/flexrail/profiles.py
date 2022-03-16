#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:35:57 2022
Sets profiles to be applied to beams and rails
@author: leonardo
"""

import numpy as np
import matplotlib.pyplot as plt

class profile(object):
    
    def __init__(self,name):
        self.name = name
        print('Profile {} created.'.format(name))
        
class planarProfile(profile):
    
    
    def __init__(self,name, fileName=''):
        super().__init__(name)
        
    def setProfilePoints(self, fileName, colDelimiter=';'):
        self.points  = np.genfromtxt(fileName, delimiter=colDelimiter)
        
        pShape = self.points.shape
        
        if pShape[1] > pShape[0]:
            self.points.tranpose()
        
    def offsetPoints(self, offset):
        self.points += offset
        
    def rotatePoints(self, angleInRad):
        R = np.array([[np.cos(angleInRad),np.sin(angleInRad)],
                      [-np.sin(angleInRad),np.cos(angleInRad)]])
        self.points = self.points.dot(R)
        
    def plotMe(self, ax = plt.gca()):
        ax.plot(self.points[:,0], self.points[:,1])
        

class wheelRailContact(object):
    
    
    def __init__(self):
        pass
        
    def setWheel(self,wheelProf):
        self.wheel = wheelProf
        
    def setRail(self,railProf):
        self.rail = railProf
        
    def searchContactPoint(self, tol = 1e-6):
        
        # PART 1: exclude points that can not participate
        # Initial supposition: wheel and rail on first quadrant
        initialPoint = self.wheel.points[0]
        if self.rail.points[0,0] > initialPoint[0]:
            initialPoint = self.rail.points[0]
        
        finalPoint = self.wheel.points[-1]
        if self.rail.points[-1,0] < finalPoint[0]:
            finalPoint = self.rail.points[-1]
        
        
        
        pass
        

if __name__=='__main__':
    w = planarProfile('wheel')
    r = planarProfile('rail')
    
    w.setProfilePoints('/home/leonardo/git/mbsim-wagons/roda.dat')
    r.setProfilePoints('/home/leonardo/git/mbsim-wagons/tr68.dat')
    
    w.offsetPoints(np.array([.340,0.685]))
    r.offsetPoints(np.array([.500,0.0]))
    r.rotatePoints(np.arctan(1./40))
    
    ax = plt.gca()
    w.plotMe(ax)
    r.plotMe(ax)
    ax.axis('equal')
    
    wr = wheelRailContact()
    wr.setWheel(w)
    wr.setRail(r)
    wr.searchContactPoint()