#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  25 10:11:22 2022

Conversion from old Python file to Cython

@author: leonardo
"""

import numpy as np

cdef class material(object):
    '''
    material
    '''
    
    def __init__(self,name):
        self.name = name
        
class linearElasticMaterial(material):
    '''
    Linear elastic material
    '''
    
    def __init__(self,name,E,nu,rho):
        ''' Material initialization method

        Parameters
        ----------
        name : STRING
            MATERIAL NAME.
        E : DOUBLE
            YOUNG'S MODULUS.
        nu : DOUBLE
            POISSON'S RATIO.
        rho : DOUBLE
            DENSITY.

        Returns
        -------
        None.

        '''
        super().__init__(name)
        
        self.E = E
        self.nu = nu
        self.rho = rho
        
        # Lamé constants for 3d stress-strain state
        self.mu = self.E / (1+self.nu) * 0.5
        self.lam = self.nu * self.E / ( (1+self.nu)*(1-2*self.nu) )
        
        D = np.zeros((6,6))
        cdef double[:,:] D_view = D
        
        # following Lai, Introduction to Continuum Mechanics, 2010 notation
        C11 = self.lam + 2*self.mu
        D[0,0] = C11
        D[1,1] = C11
        D[2,2] = C11
        
        D[0,1] = self.lam
        D[1,2] = self.lam
        D[2,3] = self.lam
        D[1,0] = self.lam
        D[2,1] = self.lam
        D[3,2] = self.lam
        
        D[3,3] = 2*self.mu
        D[4,4] = 2*self.mu
        D[5,5] = 2*self.mu
        
        self.constitutiveMatrix = D
        
        
        
    def stressTensor(self,strainTensor,bint split=False):
        '''
        Calcultes the stress tensor given a strain tensor and 
        supposing the material is linear elastic

        Parameters
        ----------
        strainTensor : MATRIX
            2nd order strain tensor.

        Returns
        -------
        2nd order stress tensor
        
        TODO (2021.08.23):
        -------
        This method should be placed inside the MATERIAL class, because it
        depends on the constitutive model. I'll wait, however, until the model
        is full 3D
        
        Reference:
        Lai, W. Michael, David Rubin, e Erhard Krempl. 
        Introduction to continuum mechanics. 
        4th ed. Amsterdam ; Boston: Butterworth-Heinemann/Elsevier, 2010. pp 208

        '''
        
        T = np.zeros_like(strainTensor,dtype=np.float64)
        Tc = T.copy()
        cdef double [:,:] T_view
        cdef double [:,:] Tc_view
       
        
        # gets Lamè constants from material
        cdef double mu = self.mu
        cdef double E = self.E
        cdef double nu = self.nu
        cdef double ks = 10*(1+nu)/(12+11*nu)   # shear constant
        
        cdef Py_ssize_t i,j
        
        
        if T.shape[0] == 2:
            if not split:
            # regular stress tensor
                for i in range(2):
                    for j in range(2):
                        T[0,0] = strainTensor[0,0] + self.nu*strainTensor[1,1]
                        T[1,1] = self.nu*strainTensor[0,0] + strainTensor[1,1] 
                        T[1,0] = (1-self.nu) * strainTensor[1,0]
                        T[0,1] = T[1,0]
                        T = self.E / (1-self.nu**2) * T
                    
                    
            # locking-free stress tensor
            else:
            
                T[0,0] = E * strainTensor[0,0]
                T[1,1] = E * strainTensor[1,1]
                T[0,1] = 2 * ks * mu * strainTensor[0,1]
                T[1,0] = T[0,1]
                
                Tc[0,0] = nu * ( nu * strainTensor[0,0] + strainTensor[1,1])
                Tc[1,1] = nu * ( strainTensor[0,0] + nu * strainTensor[1,1])
                Tc *=  E / ( 1- nu*nu )
            
        elif T.shape[0] == 3:
            if not split:
            # regular stress tensor
                for i in range(3):
                    for j in range(3):
                        T[0,0] = strainTensor[0,0] + self.nu*strainTensor[1,1]
                        T[1,1] = self.nu*strainTensor[0,0] + strainTensor[1,1] 
                        T[1,0] = (1-self.nu) * strainTensor[1,0]
                        T[0,1] = T[1,0]
                        T = self.E / (1-self.nu**2) * T
                    
                    
            # locking-free stress tensor
            else:

                T = E * strainTensor
                T[0,1:2] = 2 * ks * mu * strainTensor[0,1:2]
                T[1,2] = 2 * ks * mu * strainTensor[1,2]
                T[1,0] = T[0,1]
                T[2,1] = T[1,2]
                T[2,0] = T[0,2]
                
                Tc[0,0] = 2 * nu * strainTensor[0,0] + strainTensor[1,1] + strainTensor[2,2]
                Tc[1,1] = strainTensor[0,0] + 2 * nu * strainTensor[1,1] + strainTensor[2,2]
                Tc[2,2] = strainTensor[0,0] + strainTensor[1,1] + 2 * nu * strainTensor[2,2]
                Tc *=  E * nu / (( 1 + nu ) * (1 - 2 * nu))
                
                
        return T, Tc