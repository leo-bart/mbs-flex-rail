#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 09:31 2024

@author: leonardo
"""
from nachbagauer3Dc import node, railANCF3Dquadratic
from bodiesc import flexibleRail3D
from materialsc import linearElasticMaterial
from profiles import planarProfile
import MBS.BodyConnections.Forces
import MBS.marker
import numpy as np
import helper_funcs as hf


class flexibleTrack(object):
    """Flexible track creator."""

    def __init__(self, name, gauge, length, sleeperDistance,
                 nel, system, linearBehavior=True):
        """
        Initialize flexible track.

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        gauge : TYPE
            DESCRIPTION.
        length : TYPE
            DESCRIPTION.
        sleeperDistance : DOUBLE
            Distance between two consecutive sleepers.
        nel : INTEGER
            Number of elements per rail
        system : MultibodySystem
            System to which the track belongs
        linearBehavior : BOOL
            Turn on and off linearized element behavior

        Returns
        -------
        None.

        """
        self.name = name
        self.gauge = gauge
        self.length = length
        self.slpDist = sleeperDistance
        self.nel = nel
        self.system = system

        '''
        Material
        '''
        steel = linearElasticMaterial('Steel', E=207e09,
                                      nu=0.3,
                                      rho=7.85e3)

        '''
        Mesh
        '''
        nq = []
        nq2 = []
        totalLength = 2 * nel * sleeperDistance
        for i in range(nel+1):
            nq.append(node([totalLength * i/nel,
                            -0.5*gauge-0.039, 0.1857/2,
                           0.0, -0.9968765, -0.02499219,
                           0.0, 0.02499219, -0.99968765]))
            nq2.append(node([totalLength * i/nel,
                             0.5*gauge+0.039, 0.1857/2,
                             0.0, -0.9968765,   0.02499219,
                             0.0, -0.02499219, -0.99968765]))

        eq = []
        eq2 = []
        railHeight = 0.18575
        railWidth = 6 * 0.0254
        railCentroidHeight = 0.0805
        railBaseHeight = 2.2147e-3
        railHeadHeight = 3.2165e-3
        railBaseWidth = 135.605e-3
        railWebWidth = 23.815e-3
        railHeadWidth = 78.339e-3
        railXSecArea = 8652.0e-6

        for j in range(nel):
            eq.append(
                railANCF3Dquadratic(nq[j], nq[j+1],
                                    railHeight,
                                    railWidth,
                                    railCentroidHeight,
                                    railBaseHeight,
                                    railHeadHeight,
                                    railBaseWidth,
                                    railWebWidth,
                                    railHeadWidth,
                                    railXSecArea)
            )
            eq2.append(
                railANCF3Dquadratic(nq2[j], nq2[j+1],
                                    railHeight,
                                    railWidth,
                                    railCentroidHeight,
                                    railBaseHeight,
                                    railHeadHeight,
                                    railBaseWidth,
                                    railWebWidth,
                                    railHeadWidth,
                                    railXSecArea)
            )

        '''
        Rails
        '''
        self.leftRail = flexibleRail3D('Rail L', steel)
        self.leftRail.addElement(eq)
        if linearBehavior:
            self.leftRail.nonLinear = 'L'
        else:
            self.leftRail.nonLinear = 'NL'
        self.leftRail.assembleTangentStiffnessMatrix()

        self.rightRail = flexibleRail3D('Rail R', steel)
        self.rightRail.addElement(eq2)
        if linearBehavior:
            self.rightRail.nonLinear = 'L'
        else:
            self.rightRail.nonLinear = 'NL'
        self.rightRail.assembleTangentStiffnessMatrix()

        '''
        Profiles
        '''
        leftRailMarker = self.leftRail.addMarker(
            MBS.marker.marker('Left rail marker',
                              np.array([0.0, 0.189, -0.5*gauge-0.039]),
                              np.array([[0.0, 0.0, 1.0],
                                        [0.0, -1.0, 0.0],
                                        [1.0, 0.0, 0.0]])))
        rightRailMarker = self.rightRail.addMarker(
            MBS.marker.marker('Right rail marker',
                              np.array([0.0, 0.189, 0.5*gauge+0.039]),
                              np.array([[0.0, 0.0, 1.0],
                                        [0.0, -1.0, 0.0],
                                        [1.0, 0.0, 0.0]])))

        # add profiles to rails
        rProfL = planarProfile('Left rail', convPar=-1)
        rProfL.setProfilePointsFromFile('./tr68.pro')
        rProfR = planarProfile('Right rail', convPar=1)
        rProfR.setProfilePointsFromFile('./tr68.pro')

        self.leftRail.addProfile(rProfL, leftRailMarker)
        self.rightRail.addProfile(rProfR, rightRailMarker)

        '''
        Sleepers
        '''
        self.sleepers = MBS.BodyConnections.Forces.force('Sleepers')
        self.sleepers.connect(self.leftRail, self.rightRail)
        self.sleepers.setForceFunction(self.slpForce)
        self.sleepers.setGapFunction(self.slpGap)

        self.activeSleepers = list(range(self.leftRail.numberOfNodes))

    def setupSystem(self):
        """
        Set up the multibody system.

        Returns
        -------
        None.

        """
        sys = self.system

        sys.addBody([self.leftRail, self.rightRail])
        sys.addForce(self.sleepers)

    def activeSleepersDofs(self):
        """
        Create arrays of DOFs associated to sleepers.

        Returns
        -------
        None.

        """

        # Number of DOF for each node
        block_size = self.leftRail.elementList[0].nodes[0].q.shape[0]

        # Split globalDofs into blocks of 9 elements each
        gdof = self.leftRail.globalDof
        blocks = [gdof[i:i + block_size] for i in range(0,
                                                        len(gdof), block_size)]

        # Select the blocks corresponding to indices in activeSl
        active_blocks = [blocks[i] for i in self.activeSleepers]

        # creates an array for storing the active dofs
        #  --------> global DOF numbering
        # |0 9 18 27...
        # |1 10 19 28...
        # |2 11 20 29...
        # v
        # local DOF numbering (equals node number of DOF)
        self.leftRail.activeSleepersDof = np.array(active_blocks).transpose()

        # Number of DOF for each node
        block_size = self.rightRail.elementList[0].nodes[0].q.shape[0]

        # Split globalDofs into blocks of 9 elements each
        gdof = self.rightRail.globalDof
        blocks = [gdof[i:i + block_size] for i in range(0,
                                                        len(gdof), block_size)]

        # Select the blocks corresponding to indices in activeSl
        active_blocks = [blocks[i] for i in self.activeSleepers]

        # creates an array for storing the active dofs
        #  --------> global DOF numbering
        # |0 9 18 27...
        # |1 10 19 28...
        # |2 11 20 29...
        # v
        # local DOF numbering (equals node number of DOF)
        self.rightRail.activeSleepersDof = np.array(active_blocks).transpose()

    def slpForce(self, t, p, v, m1, m2):
        """
        Force evaluation function for sleepers.

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        m1 : TYPE
            DESCRIPTION.
        m2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        leftRail = m1.parent
        rightRail = m2.parent

        # Vertical stiffness
        # states
        leftRailVerticalDofs = list(hf.mvRow2List(leftRail.activeSleepersDof,
                                                  2))
        rightRailVerticalDofs = list(hf.mvRow2List(leftRail.activeSleepersDof,
                                                   2))

        leftDist = p[leftRailVerticalDofs]
        leftVelo = v[leftRailVerticalDofs]
        rightDist = p[rightRailVerticalDofs]
        rightVelo = v[rightRailVerticalDofs]

        f = np.zeros_like(p)
        f[leftRailVerticalDofs] = 3e6 * leftDist + 3e4 * leftVelo
        f[rightRailVerticalDofs] = 3e6 * rightDist + 3e4 * rightVelo
        # increased stiffness on rail ends
        f[leftRail.globalDof[2]] += 32 * (3e6 * leftDist[0])
        f[leftRail.globalDof[-7]] += 32 * (3e6 * leftDist[-1])
        f[rightRail.globalDof[2]] += 32 * (3e6 * rightDist[0])
        f[rightRail.globalDof[-7]] += 32 * (3e6 * rightDist[-1])

        # Lateral stiffness
        stiffness = 35e9 * 0.17 * 0.24 / self.gauge

        # states
        leftRailLateralDofs = list(hf.mvRow2List(leftRail.activeSleepersDof,
                                                 1))
        rightRailLateralDofs = list(hf.mvRow2List(leftRail.activeSleepersDof,
                                                  1))
        leftRailLateralDofs = leftRail.globalDof[1::9]
        rightRailLateralDofs = rightRail.globalDof[1::9]

        leftDist = p[leftRailLateralDofs]
        leftVelo = v[leftRailLateralDofs]
        rightDist = p[rightRailLateralDofs]
        rightVelo = v[rightRailLateralDofs]

        ds = leftDist - rightDist
        dv = leftVelo - rightVelo

        f[leftRailLateralDofs] += stiffness * \
            (ds + 0.02*dv) + 1e6 * leftDist
        f[rightRailLateralDofs] += stiffness * \
            (- ds - 0.02*dv) + 1e6 * rightDist

        # Clip and pad forces

        # Rotation stiffness
        leftZ = p[leftRail.globalDof[5::9]]
        rightZ = p[rightRail.globalDof[5::9]]
        leftZdot = v[leftRail.globalDof[5::9]]
        rightZdot = v[rightRail.globalDof[5::9]]

        clipStiffness = 4.e4

        leftClipForce = (leftZ + 0.02 * leftZdot) * clipStiffness
        rightClipForce = (rightZ + 0.02 * rightZdot) * clipStiffness

        f[leftRail.globalDof[5::9]] += leftClipForce
        f[rightRail.globalDof[5::9]] += rightClipForce

        return -f

    def slpGap(self, t, p, v, m1, m2):
        """
        Gap evaluation function for sleepers.

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        m1 : TYPE
            DESCRIPTION.
        m2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.zeros((1, 3)), np.zeros((1, 3))
