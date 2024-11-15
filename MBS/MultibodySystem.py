#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:18:40 2022

@author: leonardo
"""
import numpy as np
from assimulo.solvers import IDA
from bodiesc import rigidBody, ground
from assimulo.special_systems import Mechanical_System
import matplotlib.pyplot as plt
import helper_funcs as hf


class MultibodySystem(Mechanical_System):
    """Generic class for MBS."""

    def __init__(self, name_):
        """
        Initialize the system.

        Parameters
        ----------
        name_ : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.name = name_
        self.bodies = []
        self.forceList = []
        self.constraintList = []
        self.totalDof = 0
        self.gravity = np.zeros(3)
        self.gravity[2] = 9.81
        self.useGravity = True

        self.addBody(ground())
        self.ground = self.bodies[0]

    @property
    def constrained(self):
        """
        State wheter system is constrained or not.

        Returns
        -------
        TYPE boolean
            True for constrained systems.

        """
        return len(self.constraintList) > 0

    @property
    def n_la(self):
        """
        Number of constraint equations.

        Returns
        -------
        ncstr : integer
            Number of constraint equations.

        """
        ncstr = 0
        for cstr in self.constraintList:
            ncstr += cstr.n_la
        return ncstr

    def addBody(self, bodyList):
        """
        Increment system body list.

        Parameters
        ----------
        bodyList : body or list
            Body or list of bodies.

        Returns
        -------
        None.

        """
        if type(bodyList) is list:
            self.bodies.extend(bodyList)
            for body in bodyList:
                self.totalDof += body.totalDof
        else:
            self.bodies.append(bodyList)
            self.totalDof += bodyList.totalDof

    def addForce(self, f):
        """
        Increment system force list.

        Parameters
        ----------
        f : force or list
            Force or list of forces.

        Returns
        -------
        None.

        """
        self.forceList.append(f)

    def addConstraint(self, gC):
        """
        Increment system constraint list.

        Parameters
        ----------
        gC : constraint or list
            Constraint or list of constraint.

        Returns
        -------
        None.

        """
        self.constraintList.extend(gC)

    def excludeBody(self, bodyList):
        """
        Remove bodies from body list.

        Parameters
        ----------
        bodyList : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if bodyList is list:
            self.bodies.remove(bodyList)
            for body in bodyList:
                self.totalDof -= body.totalDof
        else:
            self.bodies.remove(bodyList)
            self.totalDof -= bodyList.totalDof

    def printSystem(self):
        """
        Print system information to prompt.

        Returns
        -------
        None.

        """
        print('Printing body list in system {}.'.format(self.name))
        print('------------------------------------')
        for b in self.bodies:
            print('Body {}: type {}, {} dof'.format(b.name, b.type, b.totalDof))
        print('Totals: {} bodies, {} dofs.\n'.format(
            len(self.bodies)-1, self.totalDof))

        print('Printing force list in system {}.'.format(self.name))
        print('------------------------------------')
        for f in self.forceList:
            print('Force {}: type {}'.format(f.name, f.type))
        print('Totals: {} forces.\n'.format(len(self.forceList)))

        print('Printing constraint list in system {}.'.format(self.name))
        print('------------------------------------')
        for c in self.constraintList:
            print('Constraint {}: type {}, {} dof'.format(c.name, c.type, c.n_la))
        print('Totals: {} constraint equations.\n'.format(self.n_la))

    def forces(self, t, p, v):
        """
        Assemble the force vector for the RHS of the equations of motion.

        Parameters
        ----------
        t : double
            Time.
        p : double array
            Positions.
        v : double array
            Velocities.

        Returns
        -------
        f : double array
            Force vector.

        """
        f = np.zeros(self.totalDof)

        for bdy in self.bodies[1:]:  # first body is ground
            # rigid body weight and inertia forces
            if bdy.type == 'Rigid body' or bdy.type == 'Wheelset':
                # weight
                f[bdy.globalDof[:3]] += bdy.mass * self.gravity
                # inertia force (minus sign because belongs to lhs)
                f[bdy.globalDof[3:]] -= bdy.hVector(p[bdy.globalDof[3:]],
                                                    v[bdy.globalDof[3:]])

            # flexible body nodal forces
            if bdy.type == 'Flexible body':
                bdy.updateVelocities(v[bdy.globalDof])
                bdy.updateDisplacements(p[bdy.globalDof])

                f[bdy.globalDof] += bdy.assembleWeightVector(g=self.gravity).squeeze() \
                    - 0.02 * bdy.assembleElasticForceVector(True).squeeze() \
                    - bdy.assembleElasticForceVector().squeeze()

        for fc in self.forceList:
            f += fc.evaluateForceFunction(t, p, v, fc.marker1, fc.marker2)

        return f

    def constr3(self, t, y):
        c = []

        for cnst in self.constraintList:
            c.extend(cnst.evaluatePositionConstraintFunction(
                t, y, self.totalDof))

        return np.asarray(c)

    def GT(self, q):
        jaco = []

        for cnst in self.constraintList:
            for ji in cnst.evaluateJacobian(q):
                jaco.append(ji)

        return np.asarray(jaco).transpose()

    def setupSystem(self, t0=0.0, printSys=True):
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
            self.mass_matrix[curDof:curDof+b.totalDof,
                             curDof:curDof+b.totalDof] = b.massMatrix

            b.globalDof = list(range(curDof, curDof+bdof))

            curDof += bdof

        self.posd0 = self.vel0
        self.veld0 = [0.]*self.totalDof

        self.t0 = t0

        if printSys:
            self.printSystem()

    def postProcess(self, tvec, pvec, vvec):
        print('\nPost processing simulation...')
        posi = pvec[:, :self.n_p]
        velo = pvec[:, self.n_p:2*self.n_p]
        lamb = pvec[:, 2*self.n_p:]
        gaps = np.zeros_like(lamb)

        # positions and velocities of the bodies
        for b in self.bodies[1:]:
            # myGlobalDofs = b.globalDof
            # b.simQ = posi[:,myGlobalDofs]
            # b.simU = velo[:,myGlobalDofs]
            b.postProcess(posi, velo)
            # b.simF = forces[:,myGlobalDofs]

        constForces = []
        for f in self.forceList:
            '''Initialize simLam vector for each constraint'''
            f.initializeOutputs(tvec)
        for i in range(len(tvec)):
            GT = self.GT(posi[i])
            constForces.append(GT.dot(lamb[i]))
            # forces.append(self.forces(tvec[i],posi[i],velo[i]))
            for b in self.bodies[1:]:
                if b.type == 'Flexible body':
                    b.updateDisplacements(b.simQ[i])
                    b.updateVelocities(b.simU[i])
        for f in self.forceList:
            f.postProcess(tvec, pvec, vvec)

        constForces = np.array(constForces)
        for cst in self.constraintList:
            cst.simLam = constForces[:, cst.body1.globalDof]

# =============================================================================
# === RAILWAY SYSTEM                                                          =
# =============================================================================


class RailwaySystem(MultibodySystem):
    """Special class for railway systems."""

    def __init__(self, name_):
        super().__init__(name_)
        self.trackList = []

    def addTrack(self, trk):
        """
        Add and activate a track

        Parameters
        ----------
        trk : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.trackList.append(trk)
        self.addBody([trk.leftRail, trk.rightRail])
        self.addForce(trk.sleepers)

    def setupSystem(self, t0=0.0, printSys=True):
        super().setupSystem(t0, printSys)

        # sleepr activation has to come after system setup because of the
        # DOF numbering of flexible nodes
        self.trackList[0].activeSleepersDofs()


if __name__ == '__main__':
    from BodyConnection.Forces import linearSpring_PtP, primitivePtPConstraint
    mbs = MultibodySystem('teste')
    mbs.gravity = np.array([0, -9.8, 0], dtype=np.float64)

    body = rigidBody('Body A')
    body2 = rigidBody('Body B')

    I = 1 * np.eye(3)

    body.setMass(1.0)
    body.setInertiaTensor(I)
    body.setPositionInitialConditions(1, 0)

    body2.setMass(0.5)
    body2.setInertiaTensor(0.5*I)
    body2.setPositionInitialConditions(0, 1.)

    mbs.addBody([body, body2])

    mola = linearSpring_PtP('Spring 1', 10.0, 0.0)
    mola.connect(body, mbs.ground)
    mola2 = linearSpring_PtP('Spring 2', 10.0, 0.0)
    mola2.connect(body, body2)

    mbs.addForce(mola)
    mbs.addForce(mola2)

    constrX = primitivePtPConstraint(
        "X", body.markers[0], [0], mbs.ground.markers[0], [0])
    mbs.addConstraint(constrX)

    mbs.setupSystem()

    problem = mbs.generate_problem('ind3')

    DAE = IDA(problem)
    DAE.report_continuously = True
    DAE.inith = 1e-6
    DAE.num_threads = 12
    DAE.suppress_alg = True

    outFreq = 10e3  # Hz
    finalTime = 4

    t, p, v = DAE.simulate(finalTime, finalTime * outFreq)

    plt.plot(t, p[:, [1, 7]])
