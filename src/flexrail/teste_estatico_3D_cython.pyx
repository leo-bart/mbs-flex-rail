#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 07:24:43 2021

@author: leonardo
"""
from nachbagauer3D import node, beamANCF3Dquadratic
from materials import linearElasticMaterial
from flexibleBody import flexibleBody3D
import numpy as np
from scipy.optimize import fsolve
from time import time
'''
TEST PROGRAM
'''

def run(long nel = 2):
    steel = linearElasticMaterial('Steel',207e3,0.3,7.85e-6)
    body = flexibleBody3D('Bar',steel)
    
    
    nq = []
    # nel = 4
    totalLength = 2000.
    for i in range(nel+1):
        nq.append(node([totalLength * i/nel,0.0,0.0
                       ,0.0,1.0,0.0,
                       0.0,0.0,1.0]))
    
    
    eq = []
    for j in range(len(nq)-1):
        eq.append(beamANCF3Dquadratic(nq[j],nq[j+1],500,100))
    
    
    g = np.matrix([[0.0,-9.81]])
    
    body.addElement(eq)
    body.assembleMassMatrix()
    
    #Kt = body.assembleTangentStiffnessMatrix()
    
    
    
    # Constraint matrix 
    simBody = body
    conDof = [0,1,2,3,4,5,6,7,8]
    gdl = simBody.totalDof
    Phi = np.zeros([len(conDof),gdl])
    
    # fixed bc
    for d in conDof:
        Phi[d,d] = 1
    
    record  = []
    Qelist=[]
    
    cdef double[:] x
    cdef double[:] lam
    cdef double[:,:] Qe
    cdef double[:,:] Qa
    
    goal = [0.] * (gdl+4)
    
    def f(z):
        
        ndof = gdl
        
        x = z[0:ndof]
        lam = z[ndof:]
        
        simBody.updateDisplacements(x)
                
        Qe = simBody.assembleElasticForceVector()
        Qelist.append(Qe.T)
        Qa = Qe*0
        Qa[-8,0] = -5.0e5 * 0.5 * 0.5 * 0.5
        
        goal[0:ndof] = (-Qe.T + Qa.T - np.dot(Phi.T,lam)).tolist()[0]
        
        goal[ndof:] = np.dot(Phi,x).tolist()
        
        #print(f"max(|F|) = {np.max(np.abs(goal)):.8e}")
        
        record.append(goal)
        
        return goal
    
    z0 = np.array([0]*(gdl+len(conDof)))
    z0[-8] =  5.0e5 * 0.5 * 0.5 * 0.5
    z0[-6] = - z0[-3] * 2000
    #z = opt.newton_krylov(f,z0,maxiter=40,f_tol=1e-4,verbose=True)
    
    ts = time()
    z, info, ier, msg = fsolve(f, z0, full_output=True, col_deriv=True)
    print(msg)
    print('Simulation took {0:1.8g} seconds'.format(time()-ts))
    
    xy = simBody.plotPositions(show=True)
    tipDisp = xy[-1,:]
    gam = np.pi/2-np.arctan2(z[gdl-5]+1,z[gdl-6])
    U = simBody.totalStrainEnergy()
    print('dx = {0:1.8e} m   | dy = {1:1.8e} m \ntheta = {2:1.8e} rad| Unorm = {3:3.5e} J'.format(-z[-18]/1000,z[-17]/1000,gam,U/1000))