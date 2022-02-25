#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTE DE FLEXÃO DINÂMICA 3D

Created on Thu Nov 18 07:02:34 2021

@author: leonardo
"""

from nachbagauer3Dc import node, beamANCF3Dquadratic
from materials import linearElasticMaterial
from flexibleBodyc import flexibleBody3D
import numpy as np
from assimulo.solvers import IDA, ODASSL
from assimulo.special_systems import Mechanical_System as ms
from time import time

steel = linearElasticMaterial('Steel',207e9,0.3,7.85e3)
body = flexibleBody3D('Bar',steel)


nq = []
nel = 4
totalLength = 2.
for i in range(nel+1):
    nq.append(node([totalLength * i/nel,0.0,0.0
                   ,0.0,1.0,0.0,
                   0.0,0.0,1.0]))


eq = []
for j in range(len(nq)-1):
    eq.append(beamANCF3Dquadratic(nq[j],nq[j+1],0.500,0.100))

body.addElement(eq)


''' ASSEMBLE SYSTEM '''

def viga_balanco():
    n_p = body.totalDof
    n_la = 9
    
    M = np.zeros([n_p,n_p])
    M[:,:] = body.assembleMassMatrix()
    
    q0 = np.array([0.]*n_p)
    u0 = np.array([0.]*n_p)
    
    def forces(t,p,v):
        '''
        Calculates the forces for the dynamical system

        Parameters
        ----------
        p : array
            positions.
        v : array
            velocities.

        Returns
        -------
        forcas : array
            forces.

        '''
              
        body.updateDisplacements(p)
        
        fel = body.assembleElasticForceVector().squeeze()
        
        body.updateDisplacements(v)
        
        fel += 0.002 * body.assembleElasticForceVector().squeeze()

        tfim = 0.02
        fel[-8] += 5.0e8 * 0.5 * 0.5 * 0.5 * t/tfim if t < tfim else 5.0e8 * 0.5 * 0.5 * 0.5
        
        return - fel
    
    def posConst(t,y):
        gC = np.zeros(n_la)
        # engaste
        gC = y[:n_la]
        
        return gC
    
    def velConst(t,y):
        gC = np.zeros(n_la)
        # engaste
        gC = y[:n_la]
        
        return gC
    
    def constJacobian(q):
        
        # jacobiana é constante
        Phi = np.zeros([n_la,q.shape[0]])
        
        Phi[:,0:n_la] = np.eye(n_la)
        
        return Phi.T
    
    return ms(n_p=n_p, forces=forces, n_la=n_la, pos0=q0, vel0=u0,
              lam0=np.zeros(n_la),
              posd0=u0,veld0=0*u0,GT=constJacobian,t0=0.0,
              mass_matrix = M,
              constr3=posConst,
              constr2=velConst)


system = viga_balanco()
problem = system.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-5
DAE.num_threads = 6
DAE.suppress_alg = True

outFreq = 10e3 # Hz
finalTime = 1

problem.res(0,problem.y0,problem.yd0)

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)

q = p[:,:system.n_p]
u = p[:,system.n_p:2*system.n_p]
lam = p[:,2*system.n_p:]