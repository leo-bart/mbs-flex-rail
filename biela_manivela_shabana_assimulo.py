#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 07:26:47 2021

@author: leonardo
"""
from nachbagauer import elementQuadratic, node
from materials import linearElasticMaterial
from flexibleBody import flexibleBody
import numpy as np
from matplotlib.pyplot import plot
from assimulo.solvers import IDA, ODASSL, Radau5DAE
from assimulo.special_systems import Mechanical_System as ms


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
nel = 2
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



omega = 150.


def biela_manivela():
    n_p = 2 + biela.totalDof
    n_la = 5
    
    

    M = np.zeros([n_p,n_p])
    M[0,0] = IzManivela
    M[1:-1,1:-1] = biela.assembleMassMatrix()
    M[-1,-1] = massaPistao
        

    
    
    q0 = np.array([0.]*(2+biela.totalDof))
    q0[-1] = Lbiela + Lmanivela
    u0 = np.array([0.]*(2+biela.totalDof))
    u0[0] = omega  # velocidade angular da manivela
    vertNodes = [i for i in range(2,biela.totalDof+1,4) ]
    for i,nodeNum in enumerate(vertNodes):
        u0[nodeNum] = omega * Lmanivela * (len(vertNodes)-1 - i)/(len(vertNodes)-1)
    
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
        
        forcaPesoManivela = massaManivela * g[1] * Lmanivela / 2 * np.cos(p[0])
        
        forcaPesoBiela = biela.assembleWeightVector(g).A1
        
        biela.updateDisplacements(p[1:])
        forcaElasticaBiela = biela.assembleElasticForceVector().A1
        
        forcas = np.copy(p) * 0
        
        forcas[0] = forcaPesoManivela
        forcas[1:-1] = forcaPesoBiela.T - forcaElasticaBiela
        
        return forcas
    def GT(q):
        Phi = np.zeros([n_la,q.shape[0]])
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
        
        return Phi.T
    
    
    def constr3(t,y):
        '''Position constraints'''
        
        q = y[0:n_p]
        u = y[n_p:2*n_p]
        
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
    
    def constr2(t,y):
        '''Velocity constraints'''
        q = y[0:n_p]
        u = y[n_p:2*n_p]
        
        gC = np.zeros(5)
        
        # PINO ENTRE MANIVELA E BIELA
        gC[0] = - Lmanivela * np.sin(q[0]) * u[0] - u[1]
        gC[1] = Lmanivela * np.cos(q[0]) * u[0] - u[2]
        
        # PINO ENTRE BIELA E PISTÃO
        gC[2] = u[-5] - u[-1]
        
        # PRISMÁTICA
        gC[3] = u[-4]
        
        # MOTOR
        gC[4] = u[0] - omega
           
        return gC
        
    
    return ms(n_p=n_p, forces=forces, n_la=n_la,pos0=q0,vel0=u0,
              lam0=np.zeros(n_la),
              posd0=u0,veld0=0*u0,GT=GT,t0=0.0,
              mass_matrix = M,
              constr3=constr3,
              constr2=constr2)



bm_sys = biela_manivela()
bm = bm_sys.generate_problem('ind3')


DAE = ODASSL(bm)
DAE.report_continuously = True
DAE.inith = 1e-3
DAE.num_threads = 4
DAE.suppress_alg = True

r0 = bm.res(0,bm.y0,bm.yd0)
t,p,v=DAE.simulate(3 * np.pi / 150, 1000)

q = p[:,:bm_sys.n_p]
u = p[:,bm_sys.n_p:2*bm_sys.n_p]
lam = p[:,2*bm_sys.n_p:]

#%%
probe=[]
for qi in q:
    
    biela.updateDisplacements(qi[1:-1])
    posi = biela.plotPositions(show=False)
    posiP = np.mean(posi,axis=0).A1
    posiA = posi[0]
    posiAB = posi[-1] - posiA
    
    Ydef = posiP - (posiA + posiAB/2)
    posiAB /= np.linalg.norm(posiAB)
    posiABrot = posiAB * np.matrix([[0,1],[-1,0]])
    Ydef = np.dot(Ydef,np.column_stack((posiAB.T,posiABrot.T)))
    
    probe.append(Ydef[0].A1)
probe=np.asarray(probe)
    