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
            
        for bdy in self.bodies[1:]:  # first body is ground
            # rigid body weight and inertia forces
            if bdy.type == 'Rigid body':
                # weight
                f[bdy.globalDof[:3]] += bdy.mass * self.gravity 
                # inertia force (minus sign because belongs to lhs)
                f[bdy.globalDof[3:]] -= bdy.hVector(v[bdy.globalDof[3:]])
            
            # flexible body nodal forces
            if bdy.type == 'Flexible body':
                bdy.updateVelocities(v[bdy.globalDof])
                f[bdy.globalDof] -= 0.002 * bdy.assembleElasticForceVector(True).squeeze()
                bdy.updateDisplacements(p[bdy.globalDof])
                f[bdy.globalDof] -= bdy.assembleElasticForceVector().squeeze()
                
                f[bdy.globalDof] += bdy.assembleWeightVector(g = self.gravity).squeeze()
            
        for fc in self.forceList:
            f += fc.evaluateForceFunction(t,p,v,fc.marker1,fc.marker2)
        
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
        # first element in self.bodies is always ground, which does not add
        # dofs to the system
        for b in self.bodies[1:]:
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
            
    def postProcess(self, tvec, pvec, vvec):
        posi = pvec[:,:self.n_p]
        velo = pvec[:,self.n_p:2*self.n_p]
        lamb = pvec[:,2*self.n_p:]
        
        constForces = []
        forces = []
        for i in range(len(tvec)):
            GT = self.GT(posi[i])
            constForces.append(GT.dot(lamb[i]))
            #forces.append(self.forces(tvec[i],posi[i],velo[i]))
            
        constForces = np.array(constForces)
        for cst  in self.constraintList:
            cst.simLam = constForces[:,cst.body1.globalDof]
        
        forces = np.array(forces)    
        for b in self.bodies[1:]:
            myGlobalDofs = b.globalDof
            b.simQ = posi[:,myGlobalDofs]
            b.simU = velo[:,myGlobalDofs]
            #b.simF = forces[:,myGlobalDofs]
            
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
        coordinate system. Is a rotation matrix.
    '''
    
    def __init__(self,name_,position=np.zeros(3),orientation=np.eye(3)):
        self.name = name_
        self.position = position
        self.orientation = orientation
            
    def setParent(self, parent):
        if parent is not None:
            self.parent = parent
            self.name = parent.name + '/' + self.name
            
    def setPosition(self,posi):
        self.position = posi
        
    def setOrientation(self,orient):
        self.orientation = orient
            
class bodyConnection(object):
    def __init__(self,name_):
        self.name = name_
        self.body1 = None
        self.body2 = None
        self.type = "Generic body connection"
        
        @property
        def simLam(self):
            return self.simLam
        @simLam.setter 
        def simLam(self, simArray):
            self.simLam = simArray
        
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
        self.dofB1 = []
        self.dofB2 = []
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
        super().connect(marker1.parent, marker2.parent,
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
        self.connect(body1 = firstMarker.parent,
                     body2 = secondMarker.parent,
                     pt1 = firstMarker.position,
                     pt2 = secondMarker.position)
        self.dofB1 = self.body1.globalDof
        self.dofB2 = self.body2.globalDof
        self.n_la = 6
        
    def evaluatePositionConstraintFunction(self, t, y, tDof):
        qGlobal = y[:tDof]
        b1dof = self.dofB1
        b2dof = self.dofB2
        
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
        b1dof = self.dofB1
        b2dof = self.dofB2
        
        I = np.diag([1.,1.,1.,1.,1.,1.])
        
        jaco[:,b1dof] = I
        if b2dof != 0:
            jaco[:,b2dof] = -I
            
        return jaco
    
class hingeJoint(constraint):
    def __init__(self, name_, firstMarker, secondMarker):
        '''
        Hinge joint class
        
        A hinge joint connects two marker such that they have the same position
        and their Z axes are aligned

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
        self.connect(body1 = firstMarker.parent,
                     body2 = secondMarker.parent,
                     pt1 = firstMarker.position,
                     pt2 = secondMarker.position)
        self.dofB1 = self.body1.globalDof[[0,1,2,5]]
        self.dofB2 = self.body2.globalDof[[0,1,2,5]]
        self.n_la = 5
        
    def evaluatePositionConstraintFunction(self, t, y, tDof):
        qGlobal = y[:tDof]
        b1dof = np.array(self.body1.globalDof)
        b2dof = np.array(self.body2.globalDof)
        
        R1 = hf.cardanRotationMatrix(qGlobal[b1dof[3:]])
        R2 = hf.cardanRotationMatrix(qGlobal[b2dof[3:]]) if len(b2dof) > 0 else None
        
        rm1 = self.marker1.position
        rm2 = self.marker2.position
        
        # fix translations
        m1Pos = qGlobal[b1dof[:3]] + R1.dot(rm1)
        if len(b2dof) != 0:
            # comes here only if body2 is not ground
            m2Pos = qGlobal[b2dof[:3]] + R2.dot(rm2)
        else:
            m2Pos = rm2
            
        # fix z axis
        m1Rot = qGlobal[b1dof[[3,4]]]
        if len(b2dof) != 0:
            m2Rot = qGlobal[b2dof[[3,4]]]
        else:
            m2Rot = [0.,0.]
            
        return np.hstack((m1Pos - m2Pos, m1Rot - m2Rot))
    
    def evaluateJacobian(self, y):
        n_la = self.n_la
        jaco = np.zeros((n_la,len(y)))
        b1dof = self.body1.globalDof
        b2dof = self.body2.globalDof
        
        a = y[b1dof[3]]
        b = y[b1dof[4]]
        c = y[b1dof[5]]
        
        rm1 = self.marker1.position
              
        drhoda = hf.cardanRotationMatrixDerivative(y[b1dof[3:]],0).dot(rm1)
        drhodb = hf.cardanRotationMatrixDerivative(y[b1dof[3:]],1).dot(rm1)
        drhodc = hf.cardanRotationMatrixDerivative(y[b1dof[3:]],2).dot(rm1)     
        
        DRHO = np.vstack((drhoda,drhodb,drhodc)).transpose()
        
        I = np.diag([1.0]*n_la)
        
        jaco[:,b1dof[:-1]] = I
        jaco[0:3,3:6] += DRHO
        if len(b2dof) != 0:
            rm2 = self.marker2.position
                  
            drhoda = hf.cardanRotationMatrixDerivative(y[self.body2.globalDof[3:]],0).dot(rm2)
            drhodb = hf.cardanRotationMatrixDerivative(y[self.body2.globalDof[3:]],1).dot(rm2)
            drhodc = hf.cardanRotationMatrixDerivative(y[self.body2.globalDof[3:]],2).dot(rm2)
            
            DRHO = np.vstack((drhoda,drhodb,drhodc)).transpose()
            
            
            jaco[0:3,b2dof[3:6]] -= DRHO
            jaco[:,b2dof[:-1]] = -I
            
        return jaco

class nodeEncastreToRigidBody(constraint):
    def __init__(self,name_,flexBody, rigidBody, nodeMarker, rigidMarker):
        super().__init__(name_)
        self.type = 'Node encastre'
        self.connect(flexBody,
                     rigidBody,
                     nodeMarker,
                     rigidMarker)
        self.dofB1 = self.marker1.parent.globalDof
        self.dofB2 = self.body2.globalDof
        self.n_la = 9
        
    def evaluatePositionConstraintFunction(self, t, y, tDof):
        qGlobal = y[:tDof]
        b1dof = np.array(self.body1.globalDof)
        b2dof = self.body2.globalDof
        myNode = self.marker1.parent
        
        # translation constraint
        if len(b2dof) > 0:
            R2 = hf.cardanRotationMatrix(qGlobal[b2dof[3:]])
            p2 = qGlobal[b2dof[:3]] + R2.dot(self.marker2.position)
        else:
            p2 = self.marker2.position
            R2 = np.eye(3)
        
        qTotal1 = qGlobal[b1dof[myNode.globalDof]] + myNode.q0
        p1 = qTotal1[:3]
        
        
        # rotation constraint
        # e12 = hf.unitaryVector(qTotal1[3:6])[0]
        # e13 = hf.unitaryVector(qTotal1[6:9])[0]
        e12 = qTotal1[3:6]
        e13 = qTotal1[6:9]
                
        g = np.hstack((p1 - p2,   # difference in position
                        e12 - R2.transpose()[1], # difference in orientation vector 1
                        e13 - R2.transpose()[2]))# difference in orientation vector 2
        
        return g
    
    def evaluateJacobian(self, y):
        n_la = self.n_la
        jaco = np.zeros((n_la,len(y)))
        
        b1dof = np.array(self.body1.globalDof)
        b2dof = np.array(self.body2.globalDof)
        myNode = self.marker1.parent
        
        I = np.eye(n_la)
        
        jaco[:,b1dof[myNode.globalDof]] = I
        if len(b2dof) > 0:
            y2 = y[b2dof[3:]]
            
            dR2da = hf.cardanRotationMatrixDerivative(y2,0)
            dR2db = hf.cardanRotationMatrixDerivative(y2,1)
            dR2dc = hf.cardanRotationMatrixDerivative(y2,2)
            
            jaco[:3,b2dof[:3]] = - np.eye(3)
            jaco[3:,b2dof[3]] = - dR2da.transpose().flatten()[3:]
            jaco[3:,b2dof[4]] = - dR2db.transpose().flatten()[3:]
            jaco[3:,b2dof[5]] = - dR2dc.transpose().flatten()[3:]
        
        return jaco
    
class nodeBallJointToRigidBody(constraint):
    def __init__(self,name_,flexBody, rigidBody, nodeMarker, rigidMarker):
        super().__init__(name_)
        self.type = 'Node ball joint'
        self.connect(flexBody,
                     rigidBody,
                     nodeMarker,
                     rigidMarker)
        self.dofB1 = self.marker1.parent.globalDof[:3]
        self.dofB2 = self.body2.globalDof[:3]
        self.n_la = 3
        
    def evaluatePositionConstraintFunction(self, t, y, tDof):
        qGlobal = y[:tDof]
        b1dof = np.array(self.body1.globalDof)
        b2dof = self.body2.globalDof
        myNode = self.marker1.parent
        
        # translation constraint
        if len(b2dof) > 0:
            R2 = hf.cardanRotationMatrix(qGlobal[b2dof[3:]])
            p2 = qGlobal[b2dof[:3]] + R2.dot(self.marker2.position)
        else:
            p2 = np.zeros(3)
            R2 = np.eye(3)
        
        qTotal1 = qGlobal[b1dof[myNode.globalDof]] + myNode.q0
        p1 = qTotal1[:3]
        
        g = p1 - p2
        
        return g
    
    def evaluateJacobian(self, y):
        n_la = self.n_la
        jaco = np.zeros((n_la,len(y)))
        
        b1dof = np.array(self.body1.globalDof)
        b2dof = np.array(self.body2.globalDof)
        myNode = self.marker1.parent
        
        I = np.eye(n_la)
        
        jaco[:,b1dof[myNode.globalDof[:3]]] = I
        if len(b2dof) > 0:            
            jaco[:3,b2dof[:3]] = - np.eye(3)
        
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