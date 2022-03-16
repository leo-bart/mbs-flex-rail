#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTE COM CARGA MÓVEL

Este teste consiste de uma viga biengastada com carga móvel que se desloca
com velocidade constante.

A seção da viga é baseada nas dimensões do trilho TR68

Created on Wed Dec  1 06:45:14 2021

@author: leonardo
"""
from nachbagauer3Dc import node, railANCF3Dquadratic
from materialsc import linearElasticMaterial
from flexibleBodyc import flexibleBody3D
import numpy as np
from assimulo.solvers import IDA, ODASSL
from assimulo.special_systems import Mechanical_System as ms
from time import time
import matplotlib.pyplot as plt

steel = linearElasticMaterial('Steel',207e9,0.3,7.85e3)
body = flexibleBody3D('Bar',steel)


nq = []
nel = 5
totalLength = 11.
for i in range(nel+1):
    nq.append(node([totalLength * i/nel,0.0,0.0
                   ,0.0,1.0,0.0,
                   0.0,0.0,1.0]))


eq = []
for j in range(nel):
    eq.append(railANCF3Dquadratic(nq[j],nq[j+1],
                                  0.18575,
                                  6*0.0254,
                                  0.0805,
                                  0.022147,
                                  0.032165,
                                  135.605e-3,
                                  23.815e-3,
                                  78.339e-3,
                                  8652.e-6))

body.addElement(eq)
K =  body.assembleTangentStiffnessMatrix()
C = 0.002*K


''' ASSEMBLE SYSTEM '''

def viga_biengastada():
    n_p = body.totalDof
    n_la = 18
    
    M = np.zeros([n_p,n_p])
    M[:,:] = body.assembleMassMatrix()
    
    q0 = np.array([0.]*n_p)
    u0 = np.array([0.]*n_p)
    
    movForce = np.array([0,1000,0])
    
    
    def forcesFullCalc(t,p,v):
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
        
        fel += 0.001 * body.assembleElasticForceVector().squeeze()
        
        # effect of moving force
        if t <= 1.:
            pos = 2.*t
            
            point = np.array([pos,0.,0.])
                
            isit = body.findElement(point)
            
            localXi = eq[isit].mapToLocalCoords(point)
            
            extForce = np.dot(movForce, eq[isit].shapeFunctionMatrix(localXi[0],localXi[1],localXi[2]))
            
            fel[eq[isit].globalDof] += extForce
        
        
        
        return - fel
    
    def forcesStiffMat(t,p,v):
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
        
        fel = np.squeeze(np.asarray(K.dot(p)))#body.assembleElasticForceVector().squeeze()
        
        #body.updateDisplacements(v)
        
        fel += np.squeeze(np.asarray(C.dot(v)))#0.001 * body.assembleElasticForceVector().squeeze()
        
        # effect of moving force
        if t <= 1.:
            pos = 5.5#2.*t
            
            point = np.array([pos,0.,0.])
                
            isit = body.findElement(point)
            
            localXi = eq[isit].mapToLocalCoords(point)
            
            extForce = np.dot(movForce, eq[isit].shapeFunctionMatrix(localXi[0],localXi[1],localXi[2]))
            
            fel[eq[isit].globalDof] += extForce
        
        
        
        return - fel
    
    def posConst(t,y):
        gC = np.zeros(n_la)
        posi = y[:n_p]
        # engaste
        gC[0:9] = posi[0:9]
        gC[9:] = posi[-9:]
        
        return gC
    
    def velConst(t,y):
        gC = np.zeros(n_la)
        vel = y[n_p:2*n_p]
        # engaste
        gC[0:9] = y[0:9]
        gC[9:] = y[-9:]
        
        return gC
    
    def constJacobian(q):
        
        # jacobiana é constante
        Phi = np.zeros([n_la,n_p])
        I = np.eye(9)
        # engaste A
        Phi[0:9,0:9] = I
        # engaste B
        Phi[9:,-9:] = I
        
        return Phi.T
    
    return ms(n_p=n_p, 
              forces=forcesStiffMat, 
              n_la=n_la, pos0=q0, vel0=u0,
              lam0=np.zeros(n_la),
              posd0=u0,veld0=0*u0,GT=constJacobian,t0=0.0,
              mass_matrix = M,
              constr3=posConst,
              constr2=velConst)


system = viga_biengastada()
problem = system.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-5
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e3 # Hz
finalTime = 1.2

problem.res(0,problem.y0,problem.yd0)

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)

q = p[:,:system.n_p]
u = p[:,system.n_p:2*system.n_p]
lam = p[:,2*system.n_p:]

# plot positions
plt.figure()
for i in np.arange(0, u.shape[0],2000):
    body.updateDisplacements(q[i])
    a = body.plotPositions()
    plt.plot(a[:,0],a[:,1], label='{:.2f} s'.format(t[i]))
plt.legend()
plt.title("Trilho TR 68")