#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:18:40 2022

@author: leonardo
"""
from nachbagauer3Dc import node, railANCF3Dquadratic
from materialsc import linearElasticMaterial
#from flexibleBodyc import flexibleBody3D
from bodiesc import rigidBody
import numpy as np
from assimulo.solvers import IDA, ODASSL
from assimulo.special_systems import Mechanical_System
from time import time
import matplotlib.pyplot as plt

class MultibodySystem(Mechanical_System):
    def __init__(self,name_):
        self.name = name_
        self.bodies = []
        self.forceList = []
        self.constraintList = []
        self.totalDof = 0
        
    @property
    def constrained(self):
        return len(self.constraintList) > 0
    
    @property
    def n_la(self):
        return len(self.constraintList)
            
    def addBody(self,bodyList):
        if type(bodyList) is list:
            self.bodies.extend(bodyList)
            for body in bodyList:
                self.totalDof += body.totalDof
        else:
            self.bodies.append(bodyList)
            self.totalDof += bodyList.totalDof
    
    def addForce(self,f):
        self.forceList.append(f)
        
    def addConstraint(self, gC):
        self.constraintList.append(gC)
    
    def excludeBody(self,bodyList):
        if bodyList is list:
            self.bodies.remove(bodyList)
            for body in bodyList:
                self.totalDof -= body.totalDof
        else:
            self.bodies.remove(bodyList)
            self.totalDof -= bodyList.totalDof
            
    def printSystem(self):
        print('Printing body list in system {}.'.format(self.name))
        print('------------------------------------')
        for b in self.bodies:
            print('Body {}: type {}, {} dof'.format(b.name,b.type,b.totalDof))
        print('Totals: {} bodies, {} dofs.\n'.format(len(self.bodies),self.totalDof))
        
        print('Printing force list in system {}.'.format(self.name))
        print('------------------------------------')
        for f in self.forceList:
            print('Force {}: type {}'.format(f.name,f.type))
        print('Totals: {} forces.\n'.format(len(self.forceList)))
        
        print('Printing constraint list in system {}.'.format(self.name))
        print('------------------------------------')
        for c in self.constraintList:
            print('Constraint {}: type {}'.format(c.name,c.type))
        print('Totals: {} constraints.\n'.format(len(self.constraintList)))
        
    def forces(self,t,p,v):
        f = np.zeros(self.totalDof)
        
        for fc in self.forceList:
            f += fc.evaluateForceFunction(t,p,v,fc.body1,fc.body2)
        
        return f
    
    def constr3(self,t,y):
        c = []
        
        for cnst in self.constraintList:
            c.append(cnst.evaluateConstraintFunction(t,y, self.totalDof))
            
        return c
    
    def GT(self,q):
        jaco = []
                
        for cnst in self.constraintList:
            jaco.append(cnst.evaluateJacobian(q))
            
        return np.asarray(jaco).transpose()
    
    
    
    def setupSystem(self, t0 = 0.0, printSys = True):
        self.n_p = self.totalDof
        self.pos0 = np.zeros(self.totalDof)
        self.vel0 = self.pos0.copy()
        self.lam0 = np.zeros(self.n_la)
        self.sw0 = None
        self.mass_matrix = np.eye(self.totalDof)
        
        curDof = 0
        for b in self.bodies:
            bdof = b.totalDof
            self.pos0[curDof:curDof+bdof] = b.q0
            self.vel0[curDof:curDof+bdof] = b.u0
            self.mass_matrix[curDof:curDof+b.totalDof,curDof:curDof+b.totalDof] = b.inertiaTensor
            
            b.globalDof = list(range(curDof, curDof+bdof))
            
            curDof += bdof
            
            
        self.posd0 = self.vel0
        self.veld0 = [0.]*self.totalDof
        
        self.t0 = t0
        
        if printSys: 
            self.printSystem()
            
class marker(object):
    def __init__(self,name_):
        self.name = name_
        self.parentBody = None
        
    def setParentBody(self, parent):
        self.parentBody = parent
            
class bodyConnection(object):
    def __init__(self,name_):
        self.name = name_
        self.body1 = None
        self.body2 = None
        self.type = "Generic body connection"
        
    def connect(self, body1, body2=None):
        self.body1 = body1
        self.body2 = body2

class force(bodyConnection):
    def __init__(self,name_='Force'):
        super().__init__(name_)
        self.type = 'Force'
        self.forceFunction = None
    
    def setForceFunction(self, f):
        self.forceFunction = f
        
    def evaluateForceFunction(self, *args):
        return self.forceFunction(*args)
    
    
class constraint(bodyConnection):
    def __init__(self,name_='Connection'):
        super().__init__(name_)
        self.type = "Generic constraint"
        self.positionConstraintFunction = None
        self.velocityConstraintFunction = None
        self.constraintJacobianFunction = None
    
    def evaluateConstraintFunction(self, *args):
        return self.positionConstraintFunction(*args)
    
    def evaluateJacobian(self, *args):
        return self.constraintJacobianFunction(*args)
        
class primitivePtPConstraint(constraint):
    def __init__(self,name_, body1, dofB1, body2 = None, dofB2 = None):
        '''
        

        Parameters
        ----------
        name_ : TYPE
            DESCRIPTION.
        body1 : TYPE
            DESCRIPTION.
        dofB1 : TYPE
            DESCRIPTION.
        body2 : TYPE, optional
            Second body connected. The default is None and means that body1 is connected to the ground.
        dofB2 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if name_ == None:
            name_ = "Primitive PtP Constraint"
        super().__init__(name_)
        super().connect(body1, body2)
        self.type = "Primitive PtP Constraint"
        self.dofB1 = dofB1
        self.dofB2 = dofB2
        
    def evaluateConstraintFunction(self, t, y, tDof):
        qGlobal = y[:tDof]
        q1 = qGlobal[self.body1.globalDof[self.dofB1]]
        if self.body2 != None:
            q2 = qGlobal[self.body2.globalDof[self.dofB2]]
        else:
            q2 = 0
            
        return q1 - q2
    
    def evaluateJacobian(self, y):
        jaco = np.zeros(len(y))
        
        jaco[self.body1.globalDof[self.dofB1]] = 1
        if self.body2 != None:
            jaco[self.body2.globalDof[self.dofB2]] = -1
            
        return jaco

            
        
            
            
            
if __name__ == '__main__':
    mbs = MultibodySystem('teste')
    
    body = rigidBody('Body A',3)
    body2 = rigidBody('Body B',2)
    
    I = 2 * np.eye(3)
    
    body.setInertiaTensor(I)
    body.setPositionInitialConditions(1,0)
    
    body2.setInertiaTensor(np.array([[0.5]]))
    body2.setPositionInitialConditions(0,-1.)
    
    mbs.addBody([body,body2])
    
    
    grav = force('Grav 1')
    grav.connect(body)
    grav2 = force('Grav 2')
    grav2.connect(body2)
    mola = force('Spring 1')
    mola.connect(body)
    mola2 = force('Spring 2')
    mola2.connect(body, body2)
    
    def f(t,p,v, b1, b2=None):
        f = np.zeros_like(p)
        f[b1.globalDof[1]] = -1
        if b2 is not None:
            f[b2.globalDof[1]] = -1
        return f
    
    def f1(t,p,v, b1, b2=None):
        f = np.zeros_like(p)
        x1 = p[b1.globalDof[1]]
        x2 = 0 if b2 == None else p[b2.globalDof[1]]
        v1 = v[b1.globalDof[1]]
        v2 = 0 if b2 == None else v[b2.globalDof[1]]
        
        f[b1.globalDof[1]] = - 2 * (x1-x2)
        f[b1.globalDof[1]] += - (v1-v2)
        
        if b2 is not None:
            f[b2.globalDof[1]] = 2 * (x1-x2)
            f[b2.globalDof[1]] += (v1-v2)
        return f
    
    grav.setForceFunction(f)
    grav2.setForceFunction(f)
    mola.setForceFunction(f1)
    mola2.setForceFunction(f1)
    
    mbs.addForce(grav)
    mbs.addForce(grav2)
    mbs.addForce(mola)
    mbs.addForce(mola2)
    
    constrX = primitivePtPConstraint("X", body, 0)
    mbs.addConstraint(constrX)
    
    mbs.setupSystem()
    
    problem = mbs.generate_problem('ind3')
    
    DAE = IDA(problem)
    DAE.report_continuously = True
    DAE.inith = 1e-6
    DAE.num_threads = 12
    DAE.suppress_alg = True

    outFreq = 10e3 # Hz
    finalTime = 20

    t,p,v=DAE.simulate(finalTime, finalTime * outFreq)