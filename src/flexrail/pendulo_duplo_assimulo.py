#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 23:18:10 2021

@author: leonardo
"""
from nachbagauer import elementQuadratic, node
from materials import linearElasticMaterial
from flexibleBody import flexibleBody
import numpy as np
from matplotlib.pyplot import plot
import assimulo.solvers as sol
import assimulo.special_systems as ss

np.seterr('raise')

'''
TESTE DOS ELEMENTOS EM PÊNDULO DUPLO COM SOLVERS ASSIMULO
'''

steel = linearElasticMaterial('Steel',207e6,0.3,7.85e3)
penA = flexibleBody('Pêndulo A',steel)
penB = flexibleBody('Pêndulo B',steel)

LpenA = 400.0e-3
LpenB = 400.0e-3

altura = 20.0e-3
largura = 30.0e-3

g = [0,-9.810]

nel = 2
nodesA = [0]*(nel*3-1)*4
nodesB = [0]*(nel*3-1)*4
for i in range(nel+1):
    nodesA[i]=node(LpenA * i/nel,0.0,0.0,1.0)
    nodesB[i]=node(LpenA + LpenB * i/nel,0.0,0.0,1.0)
    
elemA = [0]*nel
elemB = [0]*nel
for i in range(nel):
    elemA[i]=elementQuadratic(nodesA[i],nodesA[i+1],altura, largura)
    elemB[i]=elementQuadratic(nodesB[i],nodesB[i+1],altura, largura)
    
    
penA.addElement(elemA)
penB.addElement(elemB)

# número total de graus de liberdade nodais
gdlA = penA.totalDof
gdlB = penB.totalDof
gdls = gdlA + gdlB 



def penduloDuplo():
    n_p = gdls
    n_la = 4
    
    M = np.zeros([n_p,n_p])
    M[:gdlA,:gdlA] = penA.assembleMassMatrix()
    M[gdlA:,gdlA:] = penB.assembleMassMatrix()
    
    q0 = np.zeros(gdls)
    u0 = q0.copy()
    
    def forces(t,p,v):
        '''
        Forces acting on the system

        Parameters
        ----------
        t : double
            Time instant.
        p : array
            position vector.
        v : array
            velocity vector.

        Returns
        -------
        Force vector.

        '''
        
        f = np.zeros_like(p)
        
        penA.updateDisplacements(p[:gdlA])
        penB.updateDisplacements(p[gdlA:])
        
        f[:gdlA] = penA.assembleWeightVector(g) - penA.assembleElasticForceVector().T
        f[gdlA:] = penB.assembleWeightVector(g) - penB.assembleElasticForceVector().T
        
        return f
    
    
    def GT(q):
        '''
        Trasnposed constraint Jacobian matrix

        Parameters
        ----------
        q : array
            position vector.

        Returns
        -------
        matrix
            transposed of the constraint Jacobian.

        '''
        
        
        Phi = np.zeros([n_la,q.shape[0]],np.float64)
        
        # pin to ground
        Phi[0,0] = 1
        Phi[1,1] = 1
        
        # pin between links
        Phi[2,gdlA-4] = 1
        Phi[2,gdlA] = -1
        
        Phi[3,gdlA-3] = 1
        Phi[3,gdlA+1] = -1
        
        return Phi.T
    
    def posConstraints(t,y):
        '''
        Definition of the position constraints in the form
        
        gC(q) = 0

        Parameters
        ----------
        t : double
            time instant.
        y : array
            state variables vector.

        Returns
        -------
        gC : array
            constraint vector.

        '''
        
        q = y[0:n_p]
        
        gC = np.zeros(n_la)
        
        gC[0] = q[0]
        gC[1] = q[1]
        
        gC[2] = q[gdlA-4] - q[gdlA]
        gC[3] = q[gdlA-3] - q[gdlA+1]
        
        return gC
    
    
    veld0 = np.zeros(n_p)
    for i in range(1,n_p,4):
        veld0[i] = g[1]
    
    return ss.Mechanical_System(n_p=n_p, forces=forces, n_la=n_la,pos0=q0,vel0=u0,
              lam0=np.zeros(n_la),
              posd0=u0,veld0=veld0,
              GT=GT,t0=0.0,
              mass_matrix = M,
              constr3=posConstraints)


pd_sys = penduloDuplo()
pd = pd_sys.generate_problem('ind3')

DAE = sol.IDA(pd)
DAE.report_continuously = True
DAE.inith = 1e-3
DAE.num_threads = 4
DAE.suppress_alg = True

t,p,v=DAE.simulate(1., 1000)

q = p[:,:pd_sys.n_p]
u = p[:,pd_sys.n_p:2*pd_sys.n_p]
lam = p[:,2*pd_sys.n_p:]
    

    
    
    
    
    
    
    
    
    
    