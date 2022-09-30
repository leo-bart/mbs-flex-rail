# -*- coding: utf-8 -*-
"""
TESTE COM MÉTODO DE EULER SEMI-IMPLÍCITO

BURGERMEISTER, B.; ARNOLD, M.; ESTERL, B. DAE time integration for real-time
 applications in multi-body dynamics. ZAMM, v. 86, n. 10, 
 p. 759–771, 24 out. 2006. 


Created on Wed Oct 13 14:11:13 2021

@author: lbaru
"""


import numpy as np
import matplotlib.pyplot as plt

massa1 = 1.0
massa2 = 1.0
massa3 = 10.0

def spring(x,u=0):
    
    fmola1 = 30. * x[0]
    fmola2 = 10. * (x[1] - x[0])
    fmola3 = 10. * (x[2] - x[1])
    
    famort1 = 4. * u[0]
    
    f1 = - fmola1 + fmola2 - famort1
    f2 = - fmola2 + fmola3
    f3 = - fmola3
    
    return np.array([f1,f2,f3])


dt = 0.5e-3
tf = 10.

t = [0]
q = []
u = []

q0 = np.array([0.,0.,0.])
u0 = np.array([0.,3.,0.])

q.append(q0)
u.append(u0)

    
def evaluateJacobian(function,x0,xdot0=0,parameter=0):
    
    f0 = function(x0,xdot0)
    
    J = np.zeros([f0.shape[0],len(x0)])
    
    if parameter == 0:
        q = x0
    else:
        q = xdot0
    
    for i,col in enumerate(J.T):
        qsave = q[i]
        q[i] += 1e-6
        
        col[:] = (function(x0,xdot0) - f0) * 1e6
        
        q[i] = qsave
        
    return J
        
        
    
    
def integrateEulerSemiImplicit(q0,u0,dt,M,extForce,Phi,tfinal):
    
    t = [0]
    q = []
    u = []
    lam = []
    
    ndof = M.shape[0]
    ncons = Phi.shape[0]
    
    LHS = np.zeros([ndof+ncons,ndof+ncons])
    RHS = np.zeros(ndof+ncons)
    
    q.append(q0)
    u.append(u0)
    lam.append(np.zeros(ncons))
    
    Jq = evaluateJacobian(extForce,q[-1],u[-1],0)
    Ju = evaluateJacobian(extForce,q[-1],u[-1],1)
    
    while t[-1] <= tfinal:
        
        
        # explicit update for positions
        deltaq = dt*u[-1]
        q.append(q[-1] + deltaq)
        
        # assemble implicit problem
        LHS[0:ndof,0:ndof] = M - dt * Ju - dt * dt * Jq
        LHS[0:ndof,ndof:] = Phi.T
        LHS[ndof:,0:ndof] = Phi
        
        RHS[0:ndof] = dt * extForce(q[-1],u[-1]) + dt * dt * np.dot(Jq,u[-1])
        RHS[ndof:] = -np.dot(Phi,u[-1])
        
        incr = np.linalg.solve(LHS,RHS)
        du = incr[0:ndof]
        lam.append(incr[ndof:])
        
        
        # implicit update for velocities
        u.append(u[-1] + du)
        
        # jacobian update using Broyden's method
        #Jq += (extForce(q[-1],u[-1]) - extForce(q[-2],u[-2]) - np.dot(Jq,deltaq)) / (np.dot(deltaq,deltaq)) * deltaq.T
        #Ju += (extForce(q[-1],u[-1]) - extForce(q[-2],u[-2]) - np.dot(Ju,du)) / (np.dot(du,du)) * du.T
        
        t.append(t[-1] + dt)
        
    return t,q,u,lam
    
M = np.diag([massa1,massa2,massa3])
    
t,q,u,lam = integrateEulerSemiImplicit(q0, u0, dt, M, spring, np.matrix([[0,0,1]]), tf)
    
plt.subplot(3,2,1)
plt.plot(t,q)
plt.ylabel('x')
plt.subplot(3,2,2)
plt.plot(t,u)
plt.ylabel('v')
plt.legend(['massa 1','massa 2','massa 3'])
plt.subplot(3,2,3)
plt.plot(t,lam)
plt.ylabel('force')
plt.subplot(3,2,4)
plt.plot(t,np.asarray(q)[:,2])
plt.ylabel('constraint violation')

    
    
