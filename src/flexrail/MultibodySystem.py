#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:18:40 2022

@author: leonardo
"""
import numpy as np
from assimulo.solvers import IDA, ODASSL
from bodiesc import rigidBody,ground
from assimulo.special_systems import Mechanical_System
from time import time
import matplotlib.pyplot as plt
import helper_funcs as hf

class MultibodySystem(Mechanical_System):
    def __init__(self,name_):
        self.name = name_
        self.bodies = []
        self.forceList = []
        self.constraintList = []
        self.totalDof = 0
        self.gravity = np.zeros(3)
        self.useGravity = True
        
        self.addBody(ground())
        self.ground = self.bodies[0]
        
        
    @property
    def constrained(self):
        return len(self.constraintList) > 0
    
    @property
    def n_la(self):
        ncstr = 0
        for cstr in self.constraintList:
            ncstr += cstr.n_la
        return ncstr
               
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
        print('Totals: {} bodies, {} dofs.\n'.format(len(self.bodies)-1,self.totalDof))
        
        print('Printing force list in system {}.'.format(self.name))
        print('------------------------------------')
        for f in self.forceList:
            print('Force {}: type {}'.format(f.name,f.type))
        print('Totals: {} forces.\n'.format(len(self.forceList)))
        
        print('Printing constraint list in system {}.'.format(self.name))
        print('------------------------------------')
        for c in self.constraintList:
            print('Constraint {}: type {}, {} dof'.format(c.name,c.type,c.n_la))
        print('Totals: {} constraint equations.\n'.format(self.n_la))
        
    def forces(self,t,p,v):
        f = np.zeros(self.totalDof)
        
        for fc in self.forceList:
            f += fc.evaluateForceFunction(t,p,v,fc.marker1,fc.marker2)
            
        for bdy in self.bodies[1:]:  # first body is ground
            f[bdy.globalDof[:3]] += bdy.mass * self.gravity 
        
        return f
    
    def constr3(self,t,y):
        c = []
        
        for cnst in self.constraintList:
            c.extend(cnst.evaluatePositionConstraintFunction(t,y, self.totalDof))
            
        return np.asarray(c)
    
    def GT(self,q):
        jaco = []
                
        for cnst in self.constraintList:
            for ji in cnst.evaluateJacobian(q):
                jaco.append(ji)
            
        return np.asarray(jaco).transpose()
    
    
    
    def setupSystem(self, t0 = 0.0, printSys = True):
        self.n_p = self.totalDof
        self.pos0 = np.zeros(self.totalDof)
        self.vel0 = self.pos0.copy()
        self.lam0 = np.zeros(self.n_la)
        self.sw0 = None
        self.mass_matrix = np.eye(self.totalDof)
        
        curDof = 0
        # first element in self.bodies is always ground, which doe
        for b in self.bodies:
            bdof = b.totalDof
            self.pos0[curDof:curDof+bdof] = b.q0
            self.vel0[curDof:curDof+bdof] = b.u0
            self.mass_matrix[curDof:curDof+b.totalDof,curDof:curDof+b.totalDof] = b.massMatrix
            
            b.globalDof = list(range(curDof, curDof+bdof))
            
            curDof += bdof
            
            
        self.posd0 = self.vel0
        self.veld0 = [0.]*self.totalDof
        
        self.t0 = t0
        
        if printSys: 
            self.printSystem()
            
class marker(object):
    '''
    Marker class. Is created aligned to the parent body coordinate system.
    
    Parameters
    ----------
    name_ : string
        Name of the marker
    position : array, optional
        Specifies the position of the array in the parent body
    orientation : array, optional
        Specifies the orientation of the marker with respect to parent body's
        coordinate system. In Cardan angles.
    '''
    
    def __init__(self,name_,position=np.zeros(3),orientation=np.zeros(3)):
        self.name = name_
        self.position = position
        self.orientation = orientation
            
    def setParentBody(self, parent):
        if parent is not None:
            self.parentBody = parent
            self.name = parent.name + '/' + self.name
            
    def setPosition(self,posi):
        self.position = posi
            
class bodyConnection(object):
    def __init__(self,name_):
        self.name = name_
        self.body1 = None
        self.body2 = None
        self.type = "Generic body connection"
        
    def connect(self, body1, body2=None, pt1=np.zeros(3),pt2=np.zeros(3)):
        '''
        Establish body connection.

        Parameters
        ----------
        body1 : body
            First connection body. Forces will be calculated with respect to this body
        body2 : body, optional
            Second connection body. The default is None, and in this case, the connection is to ground.
        pt1 : array or marker, optional
            Connection point on first body. Coordinates are given on the body1 reference marker. 
            The default is None, and in this case, the first marker on the body markers list is considered.
            If the input is an array, creates the marker on body1.
            If the input is a marker, creates a copied marker on the same location as the input.
        pt2 : array or marker, optional
            Connection point on second body. Coordinates are given on the body2 reference marker. 
            The default is None, and in this case, the first marker on the 
            body markers list is considered (or the global origina if body2 is None).
            If the input is an array, creates the marker on body1.
            If the input is a marker, creates a copied marker on the same location as the input.

        Returns
        -------
        None.

        '''
        self.body1 = body1
        self.body2 = body2
        
        if type(pt1) is np.ndarray:
            self.marker1 = self.body1.addMarker(marker('Conn_{}_marker_A'.format(self.name)))
            self.marker1.setPosition(pt1)
        else:
            self.marker1 = pt1

        
        if body2 is not None:
            if type(pt2) is np.ndarray:
                self.marker2 = self.body2.addMarker(marker('Conn_{}_marker_B'.format(self.name)))
                self.marker2.setPosition(pt2)
            else:
                self.marker2 = pt2

class force(bodyConnection):
    def __init__(self,name_='Force'):
        super().__init__(name_)
        self.type = 'Force'
        self.forceFunction = None
        self.marker1 = None
        self.marker2 = None
    
    def setForceFunction(self, f):
        self.forceFunction = f
        
    def evaluateForceFunction(self, *args):
        return self.forceFunction(*args)

class linearSpring_PtP(force):
    '''
    Linear spring object connecting two markers
    
    After declaring the spring, you'll need to call connect to join the bodies
    
    Parameters
    ----------
    name_ : str
        Name of this object
    stiffness_ : double, optional
        Value of the spring constant. Defaults to 0.0.
    damping_ : double, optional
         Value of the damping constant. Defaults to 0.0.   
    
    '''
    
    def __init__(self,name_='Linear Spring', stiffness_ = 0.0, damping_ = 0.0):
        super().__init__(name_)
        self.k = stiffness_
        self.c = damping_
        
    @property
    def stiffness(self):
        return self.k
    @stiffness.setter
    def stiffness(self,new_stiffness):
        self.k = new_stiffness
        
    @property 
    def damping(self):
        return self.c
    @damping.setter
    def damping(self, new_damping):
        self.c = new_damping
        
    def evaluateForceFunction(self,*args):
        p = args[1]
        v = args[2]
        f = np.zeros_like(p)

        dof1 = self.body1.globalDof
        
        P1 = p[dof1[:3]] + self.marker1.position
        V1 = v[dof1[:3]]

        dof2 = self.body2.globalDof
        
        if len(dof2) > 0:
            P2 = p[dof2[:3]] + self.marker2.position
            V2 = v[dof2[:3]]
        else:
            P2 = self.marker2.position
            V2 = 0
          
        axis, dist = hf.unitaryVector(P2-P1)
        
        valueForce =  (self.k * dist + self.c * (V1 - V2).dot(axis)) * axis
        f[dof1[:3]] = valueForce
        if len(dof2) > 0:
            f[dof2[:3]] = -valueForce
        
        return f
        
    
    
class constraint(bodyConnection):
    def __init__(self,name_='Connection'):
        super().__init__(name_)
        self.type = "Generic constraint"
        self.positionConstraintFunction = None
        self.velocityConstraintFunction = None
        self.constraintJacobianFunction = None
    
    def evaluatePositionConstraintFunction(self, *args):
        return self.positionConstraintFunction(*args)
    
    def evaluateJacobian(self, *args):
        return self.constraintJacobianFunction(*args)
        
class primitivePtPConstraint(constraint):
    '''
    Primitive constraint between named DOFS of two bodies

    Parameters
    ----------
    name_ : str
        Name of the constraint.
    marker1 : marker
        first marker to connect.
    dofB1 : long
        the DoF to be connected on marker 1
    marker2 : marker
        second marker to connect
    dofB2 : long
        the DoF to be connected on marker 2


    '''
    def __init__(self,name_, marker1, dofB1, marker2, dofB2):
        
        if name_ == None:
            name_ = "Primitive PtP Constraint"
        super().__init__(name_)
        super().connect(marker1.parentBody, marker2.parentBody,
                        marker1.position, marker2.position)
        self.type = "Primitive PtP Constraint"
        self.dofB1 = dofB1
        self.dofB2 = dofB2
        self.n_la = len(dofB1)
        
    def evaluatePositionConstraintFunction(self, t, y, tDof):
        '''
        Evaluates the constraint function g(p,v) = 0
        
        This method overloads parent's

        Parameters
        ----------
        t : double
            Time.
        y : array
            State vector in the form y = [p,v,lamda], whete p are the positions of the system, v the velocities and lambda are the constraint forces
        tDof : long
            Total number of DOFs in the system. This is necessary to split the state vector into positions and velocities

        Returns
        -------
        array
            DESCRIPTION.

        '''
        qGlobal = y[:tDof]
        q1 = qGlobal[self.body1.globalDof[:3]] + hf.cardanRotationMatrix(qGlobal[self.body1.globalDof[3:]]).dot(self.marker1.position)
        q1 = q1[self.dofB1]
        if self.body2.totalDof != 0:
            q2 = qGlobal[self.body2.globalDof[:3]] + hf.cardanRotationMatrix(qGlobal[self.body2.globalDof[3:]]).dot(self.marker2.position)
            q2 = q2[self.dofB2]
        else:
            q2 = np.zeros_like(q1)
            
        return (q1-q2).tolist()
    
    def evaluateJacobian(self, y):
        # TODO implement rotation part of the jacobian
        n_la = self.n_la
        jaco = np.zeros((n_la,len(y)))
        
        Rdev = np.array([-np.sin(y[self.body1.globalDof[-1]]),
                              np.cos(y[self.body1.globalDof[-1]])]) * self.marker1.position[0]
        
        for i in range(n_la):
            jaco[i,self.body1.globalDof[self.dofB1[i]]] = 1
            jaco[i,self.body1.globalDof[-1]] = Rdev[i]
            if self.body2.totalDof != 0:
                jaco[i,self.body2.globalDof[self.dofB2[i]]] = -1
            
        return jaco
    
class fixedJoint(constraint):
    def __init__(self, name_, firstMarker, secondMarker):
        '''
        Fixed joint class
        
        A fixed joint guarantees the connected markers are in the same position
        and orientation. Consequently, both markers will hold togheter, meaning
        they can not have an initial offset

        Parameters
        ----------
        name_ : str
            Joint name.
        firstMarker : marker
            Marker on first body.
        secondMarker : marker
            Marker on second body. If ground is involved, it must be the second body

        Returns
        -------
        None.

        '''
        super().__init__(name_)
        self.type = 'Fixed joint'
        self.connect(body1 = firstMarker.parentBody,
                     body2 = secondMarker.parentBody,
                     pt1 = firstMarker.position,
                     pt2 = secondMarker.position)
        self.n_la = 6
        
    def evaluatePositionConstraintFunction(self, t, y, tDof):
        qGlobal = y[:tDof]
        b1dof = self.body1.globalDof
        b2dof = self.body2.globalDof
        
        R1 = hf.cardanRotationMatrix(qGlobal[b1dof[3:]])
        R2 = hf.cardanRotationMatrix(qGlobal[b2dof[3:]])
        
        rm1 = self.marker1.position
        rm2 = self.marker2.position
        
        # fix translations
        m1Pos = qGlobal[b1dof[:3]] + R1.dot(rm1)
        if b2dof != 0:
            # comes here only if body2 is not ground
            m2Pos = qGlobal[b2dof[:3]] + R2.dot(rm2)
        else:
            m2Pos = rm2
            
        # fix rotations
        m1Rot = qGlobal[b1dof[3:]]
        if b2dof!= 0:
            m2Rot = qGlobal[b2dof[3:]]
        else:
            m2Rot = np.array(3)
            
        return np.hstack((m1Pos - m2Pos, m1Rot - m2Rot))
    
    def evaluateJacobian(self, y):
        n_la = self.n_la
        jaco = np.zeros((n_la,len(y)))
        b1dof = self.body1.globalDof
        b2dof = self.body2.globalDof
        
        I = np.diag([1.,1.,1.,1.,1.,1.])
        
        jaco[:,b1dof] = I
        if b2dof != 0:
            jaco[:,b2dof] = -I
            
        return jaco
        
        
        
            
        
            
            
            
if __name__ == '__main__':
    mbs = MultibodySystem('teste')
    mbs.gravity = np.array([0,-9.8,0],dtype=np.float64)
    
    body = rigidBody('Body A')
    body2 = rigidBody('Body B')
    
    I = 1 * np.eye(3)
    
    body.setMass(1.0)
    body.setInertiaTensor(I)
    body.setPositionInitialConditions(1,0)
    
    body2.setMass(0.5)
    body2.setInertiaTensor(0.5*I)
    body2.setPositionInitialConditions(0,1.)
    
    mbs.addBody([body,body2])
    
    
    mola = linearSpring_PtP('Spring 1',10.0,0.0)
    mola.connect(body,mbs.ground)
    mola2 = linearSpring_PtP('Spring 2',10.0,0.0)
    mola2.connect(body, body2)
    
    mbs.addForce(mola)
    mbs.addForce(mola2)
    
    constrX = primitivePtPConstraint("X", body.markers[0], [0], mbs.ground.markers[0], [0])
    mbs.addConstraint(constrX)
    
    mbs.setupSystem()
    
    problem = mbs.generate_problem('ind3')
    
    DAE = IDA(problem)
    DAE.report_continuously = True
    DAE.inith = 1e-6
    DAE.num_threads = 12
    DAE.suppress_alg = True

    outFreq = 10e3 # Hz
    finalTime = 4

    t,p,v=DAE.simulate(finalTime, finalTime * outFreq)
    
    plt.plot(t,p[:,[1,7]])