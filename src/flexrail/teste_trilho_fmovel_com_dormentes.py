#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTE COM CARGA MÓVEL

Teste com trilho cujas extremidades são livres para deslocar-se verticalmente. Com dormentes.

A seção da viga é baseada nas dimensões do trilho TR68

Created on Thi Mar 09 2022

@author: leonardo
"""
from nachbagauer3Dc import node, railANCF3Dquadratic
from materialsc import linearElasticMaterial
from bodiesc import flexibleBody3D
import numpy as np
from assimulo.solvers import IDA, ODASSL
from assimulo.special_systems import Mechanical_System
from time import time
import matplotlib.pyplot as plt

steel = linearElasticMaterial('Steel',207e9,0.3,7.85e3)
rail = flexibleBody3D('Trilho',steel)


nq = []
nel = 12
totalLength = nel*0.58*2
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

rail.addElement(eq)



''' ASSEMBLE SYSTEM '''
class multibodySystem(Mechanical_System):
    def __init__(self,name_):
        self.name = name_
        self.bodies = []
        self.totalDof
        
    def addBody(self,bodyList):
        if bodyList is list:
            self.bodies.extend(bodyList)
            for body in bodyList:
                self.totalDof += body.totalDof
        else:
            self.bodies.append(bodyList)
            self.totalDof += body.totalDof

class rigidBody(object):
    def __init__(self,name_,numberOfDof=0):
        self.name = name_
        self.totalDof = numberOfDof
        self.inertiaTensor = np.eye(numberOfDof)
        


def viga_biengastada():
    n_p = rail.totalDof
    
    constrainedDof = [0,2,3,4,5,6,7,8,
                      n_p-9,n_p-8,n_p-7,n_p-6,n_p-5,n_p-4,
                      n_p-3,n_p-2,n_p-1]
    n_la = len(constrainedDof)
    
    
    M = np.zeros([n_p,n_p])
    M[:,:] = rail.assembleMassMatrix()
    
    K =  rail.assembleTangentStiffnessMatrix()
    C = 0.02*K
    
    q0 = np.array([0.]*n_p)
    u0 = np.array([0.]*n_p)
    
    sleeperDist = 0.58
    sleeperStiff = 3.0e6
    
    movForce = np.array([0,6000,0])
    
    
    # espaçamento dos dormentes
    xSleep = np.arange(0.0,totalLength,sleeperDist)
    nSleepers = xSleep.shape[0]
    eleWithSleepers = []
    sleeperPositionOnElements = []
    for sleeperX in xSleep:
        slpPt = np.array([sleeperX,0.0,0.0])
        isit = rail.findElement(slpPt)   # placing sleepers on center line
        if isit >= 0:
            eleWithSleepers.append(isit)
            sleeperPositionOnElements.append(eq[isit].mapToLocalCoords(slpPt))
    sleeperForce = np.array([0.,sleeperStiff,0.])
            
    def sleeperForces():
        f = np.zeros(rail.totalDof)
        
        for i in range(len(eleWithSleepers)):
            ele = eq[eleWithSleepers[i]]
            posi = sleeperPositionOnElements[i]
            dy = ele.interpolatePosition(posi[0],
                                          posi[1],
                                          posi[2])[1]
            slpF = -dy*sleeperForce if dy < 0 else 0*sleeperForce
            f[ele.globalDof] += np.dot(slpF, ele.shapeFunctionMatrix(posi[0],
                                          posi[1],
                                          posi[2]))
        
    
        return f
    
    
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
              
        rail.updateDisplacements(p)
        
        fel = rail.assembleElasticForceVector().squeeze()
        
        rail.updateDisplacements(v)
        
        fel += 0.002 * rail.assembleElasticForceVector().squeeze()
        
        # effect of moving force
        if t <= 1.:
            pos = totalLength*t
            
            point = np.array([pos,0.,0.])
                
            isit = rail.findElement(point)
            
            localXi = eq[isit].mapToLocalCoords(point)
            
            extForce = np.dot(movForce, eq[isit].shapeFunctionMatrix(localXi[0],localXi[1],localXi[2]))
            
            fel[eq[isit].globalDof] += extForce
            
        
        #fel -= sleeperForces()
        
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
              
        rail.updateDisplacements(p)
        
        fel = np.squeeze(np.asarray(K.dot(p)))#body.assembleElasticForceVector().squeeze()
        
        #body.updateDisplacements(v)
        
        fel += np.squeeze(np.asarray(C.dot(v)))#0.001 * body.assembleElasticForceVector().squeeze()
        
        # effect of moving force
        if t <= 1.:
            pos = totalLength*t
            
            point = np.array([pos,0.09,0.])
                
            isit = rail.findElement(point)
            
            localXi = eq[isit].mapToLocalCoords(point)
            
            extForce = np.dot(movForce, eq[isit].shapeFunctionMatrix(localXi[0],localXi[1],localXi[2]))
            
            fel[eq[isit].globalDof] += extForce
        
        fel -= sleeperForces()
        
        return - fel
    
    def posConst(t,y):
        gC = np.zeros(n_la)
        posi = y[:n_p]
         
        for i in range(n_la):
            gC[i] = posi[constrainedDof[i]]
        
        
        
        return gC
    
    def velConst(t,y):
        gC = np.zeros(n_la)
        vel = y[n_p:2*n_p]
        
        for i in range(n_la):
            gC[i] = vel[constrainedDof[i]]
        
        return gC
    
    def constJacobian(q):
        
        # jacobiana é constante
        Phi = np.zeros([n_la,n_p])
        for i in range(n_la):
            Phi[i,constrainedDof[i]] = 1.
        
        return Phi.T
    
    return Mechanical_System(
        n_p=n_p, 
        forces=forcesStiffMat, 
        n_la=n_la, pos0=q0, vel0=u0,
        lam0=np.zeros(n_la),
        posd0=u0,veld0=0*u0,GT=constJacobian,t0=0.0,
        mass_matrix = M,
        constr3=posConst,
        constr2=velConst
        )


system = viga_biengastada()
problem = system.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-5
DAE.maxh = 1e-4
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e3 # Hz
finalTime = 2.0

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)

q = p[:,:system.n_p]
u = p[:,system.n_p:2*system.n_p]
lam = p[:,2*system.n_p:]

# plot positions
plt.figure()
nplots = 4
for i in np.arange(0, u.shape[0],int(u.shape[0]/nplots)):
    rail.updateDisplacements(q[i])
    a = rail.plotPositions(30)
    plt.plot(a[:,0],a[:,1], color='{}'.format(1-t[i]/t[-1]),label='{:.2f} s'.format(t[i]))
plt.legend()
plt.title("Trilho TR 68")