#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 14:27:00 2021

@author: leonardo
"""

import numpy as np

def evaluateJacobian(function,x0,xdot0=0,parameter=0,dofCalculate=None):
    
       
    f0 = function(x0,xdot0)
    
    if dofCalculate is None:
        dofCalculate = range(f0.shape[0])
    
    J = np.zeros([f0.shape[0],x0.shape[0]])
    
    if parameter == 0:
        q = x0
    else:
        q = xdot0
    
    for i in dofCalculate:
        qsave = q[i]
        q[i] += 1e-6
        
        J[:,i] = (function(x0,xdot0) - f0) * 1e6
        
        q[i] = qsave
    return J

def integrateEulerSemiImplicit(q0,u0,dt,M,extForce,constrFunc,
                               Phi,tfinal,writeOutput=None,
                               dtout=1e-3, maxStep=1e-3, minStep=1e-7):
    
    tout = 0.
    
    tn = 0.
    qn = q0.copy()
    un = u0.copy()
    muC = 0 * constrFunc(qn,un,tn) # position constraint Lagrange multipliers
    
    # constraint Jacobian
    Cqn, wn = Phi(qn,un)
    
    ndof = M.shape[0]
    ncons = Cqn.shape[0]
    
    LHS = np.zeros([ndof+ncons,ndof+ncons])
    RHS = np.zeros(ndof+ncons)
    
    Minv = np.linalg.inv(M)
    
    writeOutput(tout,q0,u0,np.zeros(ncons))
   
    
    Jq = evaluateJacobian(extForce,qn,un,0)
    Ju = evaluateJacobian(extForce,qn,un,1)
   
    jacoCounter = 0
    
    def constraintCalculator(muc_):
        #TODO deal with velocity constraints at position level
        
        return constrFunc(qnp1 + dt*un - dt*np.dot(np.dot(Minv,Phi(qn,un)[0].T),muc_),un,tn)
    
    while tn <=tfinal:
        
        # constraint Jacobian
        Cqn,wn = Phi(qn,un)
        
        # explicit update for positions
        deltaq = dt*un - dt*np.dot(np.dot(Minv,Cqn.T),muC)
        qnp1 = qn + deltaq
        
        #muC, info,i,r = fsolve(constraintCalculator,muC,full_output=True,col_deriv=True)
        
        # updated constraint Jacobian
        Cqnp1,wnp1 = Phi(qnp1,un)
        
        # assemble implicit problem
        LHS[0:ndof,0:ndof] = M - dt * Ju - dt * dt * Jq
        if Cqn.shape[0] != 0:
            LHS[0:ndof,ndof:] = Cqn.T
            LHS[ndof:,0:ndof] = Cqnp1
        
        RHS[0:ndof] = dt * extForce(qn,un) + dt * dt * np.dot(Jq,un)
        if Cqn.shape[0] != 0:
            RHS[ndof:] = -np.dot(Cqnp1,un) + wnp1
        
        incr = np.linalg.solve(LHS,RHS)
        du = incr[0:ndof]
        lamnp1 = incr[ndof:]/dt
        
        
        # implicit update for velocities
        unp1 = un + du
        
        # jacobian update using Broyden's method
        df = extForce(qnp1,unp1) - extForce(qn,un)
        dJq = np.outer((df - np.dot(Jq,deltaq)) / (np.dot(deltaq,deltaq)+1e-18), deltaq)
        
        indicador = np.linalg.norm(dJq)  / (np.linalg.norm(Jq)+1e-18)
        
        if jacoCounter < 10 and indicador < 1e-4:
            jacoCounter = 0
            Jq += dJq 
            #Ju += (df - np.dot(Ju,du)) / (np.dot(du,du)+2e-16) * du.T
        else:
            Jq = evaluateJacobian(extForce,qnp1,unp1,0)
            #Ju = 0.0002*Jq
            #Ju = evaluateJacobian(extForce,q[-1],u[-1],1)
            if jacoCounter > 5:
                # if the number of steps without jacobian reevaluations is smaller than N, reduce time step
                dt = max(dt/2,minStep)
                #print('Reduced time step to {0:1.4e}'.format(dt))
            #print('Reevaluate jacobian after {0} steps at {1:1.8f} s'.format(jacoCounter,tn))
            jacoCounter += 1

        if jacoCounter > 8:
                dt = min(1.5 * dt,maxStep)
                #print('Increased time step to {0:1.4e}'.format(dt))
                
        
        tnp1 = tn + dt
        #print(dt)
        
        if tnp1 - tout > dtout:
            writeOutput(tnp1,qnp1,unp1,lamnp1)
            print(dt,indicador)
            tout = tnp1
            
            
        # updates states
        qn = qnp1
        un = unp1
        tn = tnp1