#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:30:48 2023

@author: leonardo
"""
import MBS.BodyConnections.BodyConnection
import numpy as np
import helper_funcs as hf

class force(MBS.BodyConnections.BodyConnection.bodyConnection):
    def __init__(self,name_='Force'):
        super().__init__(name_)
        self.type = 'Force'
        self.forceFunction = None
        self.marker1 = None
        self.marker2 = None
    
    def setForceFunction(self, f):
        self.forceFunction = f
        
    def evaluateForceFunction(self, *args):
        return self.forceFunction(*args)

class linearSpring_PtP(force):
    """
    Linear spring object connecting two markers
    
    After declaring the spring, you'll need to call connect to join the bodies
    
    Parameters
    ----------
    name_ : str
        Name of this object
    stiffness_ : double, optional
        Value of the spring constant. Defaults to 0.0.
    damping_ : double, optional
         Value of the damping constant. Defaults to 0.0.   
    
    """
    
    def __init__(self,name_='Linear Spring', stiffness_ = 0.0, damping_ = 0.0):
        super().__init__(name_)
        self.k = stiffness_
        self.c = damping_
        
    @property
    def stiffness(self):
        return self.k
    @stiffness.setter
    def stiffness(self,new_stiffness):
        self.k = new_stiffness
        
    @property 
    def damping(self):
        return self.c
    @damping.setter
    def damping(self, new_damping):
        self.c = new_damping
        
    def evaluateForceFunction(self,*args):
        p = args[1]
        v = args[2]
        f = np.zeros_like(p)

        dof1 = self.body1.globalDof
        
        P1 = p[dof1[:3]] + self.marker1.position
        V1 = v[dof1[:3]]
        
        dof2 = self.body2.globalDof
        
        if len(dof2) > 0:
            P2 = p[dof2[:3]] + self.marker2.position
            V2 = v[dof2[:3]]
        else:
            P2 = self.marker2.position
            V2 = 0
          
        axis, dist = hf.unitaryVector(P2-P1)
        
        valueForce =  (self.k * dist + self.c * (V1 - V2).dot(axis)) * axis
        f[dof1[:3]] = valueForce
        if len(dof2) > 0:
            f[dof2[:3]] = -valueForce
        
        return f