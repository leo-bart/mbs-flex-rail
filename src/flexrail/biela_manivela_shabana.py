#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 07:36:03 2021

@author: leonardo

Constraint stabilization using Gear method


"""

from nachbagauer import elementQuadratic, node
from materials import linearElasticMaterial
from flexibleBody import flexibleBody
import numpy as np
from solver import integrateEulerSemiImplicit
from matplotlib import pyplot as plt
from scipy.optimize import fsolve


np.seterr('raise')

'''
TESTE DOS ELEMENTOS EM BIELA MANIVELA
'''



steel = linearElasticMaterial('Steel',200e9,0.3,7.85e3)
biela = flexibleBody('Biela',steel)

# comprimentos
Lmanivela = 150.0e-3
Lbiela = 300.0e-3

# seção transversal da biela
diametro = 6.0e-3

# gravidade
g = [0,-9.810*0]

# malha
nel = 4
nodesA = [0]*(nel*3-1)
for i in range(nel+1):
    nodesA[i]=node(Lmanivela + Lbiela * i/nel,0.0,0.0,1.0)
    
elemA = [0]*nel
for i in range(nel):
    elemA[i]=elementQuadratic(nodesA[i],nodesA[i+1], 0.876*diametro, 0.876*diametro)
    
    
biela.addElement(elemA)


'''
Manivela
'''

massaManivela = np.pi * diametro**2 / 4 * Lmanivela * steel.rho
IzManivela = massaManivela * (0.25  * diametro**2 / 4 + 0.334  * Lmanivela**2)

'''
Pistão
'''

massaPistao = massaManivela



'''
RESTRIÇõES
'''

def constraints(q,u,t):
    
    gC = np.zeros(5)
    
    # PINO ENTRE MANIVELA E BIELA
    gC[0] = Lmanivela * np.cos(q[0]) - q[1] - Lmanivela
    gC[1] = Lmanivela * np.sin(q[0]) - q[2]
    
    # PINO ENTRE BIELA E PISTÃO
    gC[2] = q[-5] - q[-1] + (Lmanivela + Lbiela)
    
    # PRISMÁTICA
    gC[3] = q[-4]
    
    # MOTOR
    gC[4] = q[0] - u[0]*t
       
    return gC

def Phi(q,u):
    
    Phi = np.zeros([5,q.shape[0]])
    w = np.zeros(Phi.shape[0])
    
    # PINO ENTRE MANIVELA E BIELA
    Phi[0,0] = Lmanivela*np.sin(q[0])
    Phi[1,0] = -Lmanivela*np.cos(q[0])
    Phi[0,1] = 1.
    Phi[1,2] = 1.
    
    # PINO ENTRE BIELA E PISTÃO
    Phi[2,-5] = 1.
    Phi[2,-1] = -1.
    
    # PRISMÁTICA
    Phi[3,-4] = 1.
    
    # VELOCIDADE ANGULAR CONSTANTE
    Phi[4,0] = 1.
    w[4] = 150
    
    return Phi, w

def forcas(q,u):
    '''
    Calculates the forces for the dynamical system

    Parameters
    ----------
    q : array
        positions.
    u : array
        velocities.

    Returns
    -------
    forcas : array
        forces.

    '''
    
    forcaPesoManivela = massaManivela * g[1] * Lmanivela / 2 * np.cos(q[0])
    
    forcaPesoBiela = biela.assembleWeightVector(g).A1
    
    biela.updateDisplacements(q[1:])
    forcaElasticaBiela = biela.assembleElasticForceVector().A1
    
    forcas = np.copy(q) * 0
    
    forcas[0] = forcaPesoManivela
    forcas[1:-1] = forcaPesoBiela.T - forcaElasticaBiela
    
    return forcas
'''
SOLVER
'''

omega = 150


t = [0]
q = []
u = []

q0 = np.array([0.]*(2+biela.totalDof))
q0[-1] = Lbiela + Lmanivela
u0 = np.array([0.]*(2+biela.totalDof))
u0[0] = omega  # velocidade angular da manivela
u0[2] = omega * Lmanivela
u0[6] = omega * Lmanivela * 0.875
u0[10] = omega * Lmanivela * 0.75
u0[14] = omega * Lmanivela * 0.625
u0[18] = omega * Lmanivela * 0.500
u0[22] = omega * Lmanivela * 0.375
u0[26] = omega * Lmanivela * 0.250
u0[30] = omega * Lmanivela * 0.125

q.append(q0)
u.append(u0)









t = []
q = []
u = []
lam = []
probe = []
U = []

def output(t_,q_,u_,lam_):
    t.append(t_)
    q.append(q_)
    u.append(u_)
    lam.append(lam_)
    
    posi = biela.plotPositions(show=False)
    posiP = np.mean(posi,axis=0).A1
    posiA = posi[0]
    posiAB = posi[-1] - posiA
    
    Ydef = posiP - (posiA + posiAB/2)
    posiAB /= np.linalg.norm(posiAB)
    posiABrot = posiAB * np.matrix([[0,1],[-1,0]])
    Ydef = np.dot(Ydef,np.column_stack((posiAB.T,posiABrot.T)))
    
    probe.append(Ydef[0].A1)
    
    
    print('t = {0:1.8f}'.format(t_))
    




            
    
M = np.zeros([2+biela.totalDof,2+biela.totalDof])
M[0,0] = IzManivela
M[1:-1,1:-1] = biela.assembleMassMatrix()
M[-1,-1] = massaPistao
 

dt = 5e-6
tf = 3 * np.pi / 150
   
integrateEulerSemiImplicit(q0, u0, dt, M, 
                           forcas, constraints, Phi,
                           tf, output, dtout=2e-4, minStep=10.e-7, maxStep=2e-5)

q = np.asmatrix(q)
u = np.asmatrix(u)
probe = np.asmatrix(probe)


