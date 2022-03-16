#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:55:42 2022

Classes to use with Multibody System as bodies

@author: leonardo
"""

import numpy as np

cdef class body(object):
    
    cdef str name, type
    cdef int totalDof
    cdef double[:,:] inertiaTensor
    cdef double[:] q0, u0
    cdef public list globalDof
    
    def __init__(self,name_,numberOfDof=0):
        self.name = name_
        self.type = 'Generic body descriptor'
        self.totalDof = numberOfDof
        self.inertiaTensor = np.eye(numberOfDof)
        self.q0 = np.zeros(self.totalDof, dtype=np.float64)
        self.u0 = self.q0.copy()
        self.globalDof = []
    
    @property
    def name(self):
        return self.name
    
    @property
    def type(self):
        return self.type
    
    @property
    def totalDof(self):
        return self.totalDof
    
    @property
    def q0(self):
        return np.array(self.q0)
    
    @property
    def u0(self):
        return np.array(self.u0)    
    
    @property
    def inertiaTensor(self):
        return np.array(self.inertiaTensor)
    
    
    
    ####### METHODS ##################################
    def setPositionInitialConditions(self,*args):
        '''
        Set the initial conditions on position level
        
        This function has two possible calls
        
        setPositionInitialConditions(q) expects q to be an array with all dofs specified
        
        setPositionInitialConditions(qInd,val) expects qInd to be the index of the dof and val is the initial value attributed to q[qInd].

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if len(args) == 1:
            if args[0].size == self.totalDof:
                # then the input vector has the correct size
                self.q0 = args[0]
            else:
                print('Body {}: error on initial conditions attribution: expected a {}-dimensional vector'.format(self.name,self.totalDof))
        elif len(args) == 2:
            self.q0[args[0]] = args[1]
        else:
            print('{} setInitialConditions: expected 1 or 2 elements.'.format(self.name))
    
    def setVelocityInitialConditions(self,*args):
        '''
        Set the initial conditions on position level
        
        This function has two possible calls
        
        setPositionInitialConditions(q) expects q to be an array with all dofs specified
        
        setPositionInitialConditions(qInd,val) expects qInd to be the index of the dof and val is the initial value attributed to q[qInd].

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if len(args) == 1:
            if args[0].size == self.totalDof:
                # then the input vector has the correct size
                self.u0 = args[0]
            else:
                print('Body {}: error on initial conditions attribution: expected a {}-dimensional vector'.format(self.name,self.totalDof))
        elif len(args) == 2:
            self.u0[args[0]] = args[1]
        else:
            print('{} setInitialConditions: expected 1 or 2 elements.'.format(self.name))

cdef class rigidBody(body):
    def __init__(self,name_,numberOfDof=6):        
        super().__init__(name_,numberOfDof)
        self.type = 'Rigid body'
        
    def setDof(self,freeDof):
        self.totalDof = freeDof
        self.inertiaTensor = np.eye(freeDof)
        self.q0.resize(self.totalDof)
        self.u0.resize(self.totalDof)
        
    def setInertiaTensor(self, tensor):
        if tensor.shape == (self.totalDof, self.totalDof):
            self.inertiaTensor = tensor
        else:
            print('Error in inertia attribution. Need a {0}x{0} tensor.'.format([self.totalDof]))
    
    