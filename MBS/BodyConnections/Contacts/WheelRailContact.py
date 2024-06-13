#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wheel-rail contact force definition

Created on Sun Oct  1 11:19:18 2023

@author: leonardo
"""

import MBS.BodyConnections.Contacts.Contact
import numpy as np
import helper_funcs as hf
import matplotlib.pyplot as plt
import gjk
import gjkc
import polachContactForces as pcf
from profiles import planarProfile


class wrContact (MBS.BodyConnections.Contacts.Contact.contact):
    """
    New version of WR contact.

    This version receives as input three bodies: the wheelset and two rails.
    Other inputs are: track gauge, face to face wheel distance, and nominal
    wheel radius
    """

    def __init__(self,
                 _leftRail_, _rightRail_,
                 _wheelset_,
                 _name_='Wheel rail contact'):
        """
        Class constructor.

        Parameters
        ----------
        _leftRail_ : body
            Left rail body.
        _rightRail_ : body
            Right rail body.
        _wheelset_ : wheelset.
            Wheelset body.
        _name_ : str, optional
            Contact name. The default is 'Wheel rail contact'.

        Returns
        -------
        None.

        """
        super().__init__(_name_)

        self.leftRail = _leftRail_
        self.rightRail = _rightRail_
        self.wheelset = _wheelset_

    def evaluateForceFunction(self, t, p, v, *args):
        """
        Calculate contact force.

        Parameters
        ----------
        t : double
            Time.
        p : double vector array
            State vector.
        v : double vector array
            Time derivative of state vector.
        *args : Variable number of positional arguments.

        Returns
        -------
        None.

        """
        # wheelset reference marker position
        wstPosition = p[self.wheelset.globalDof]
        wstPosition[4] = 0
        wstVelocity = v[self.wheelset.globalDof]
        # gets wheel Cardan angles and removes rotation around shaft axis
        wstCardans = wstPosition[3:]
        # gets wheelset rotation matrix
        Rwst = hf.cardanRotationMatrix(wstCardans)

        cNormalLeft, cPointsLeft, gapLeft, elemLeft = \
            self.evaluateGapFunction(t, p, v, 'left')
        cNormalRight, cPointsRight, gapRight, elemRight = \
            self.evaluateGapFunction(t, p, v, 'right')

        f = np.zeros_like(p)

        if gapLeft < 0.0:
            self.calculateContactForce(Rwst, cNormalLeft, gapLeft,
                                       cPointsLeft['wheel'],
                                       cPointsLeft['rail'], wstPosition,
                                       wstVelocity,
                                       elemLeft, f)
        if gapRight < 0.0:
            self.calculateContactForce(Rwst, cNormalRight, gapRight,
                                       cPointsRight['wheel'],
                                       cPointsRight['rail'], wstPosition,
                                       wstVelocity,
                                       elemRight, f)

        return f

    def calculateContactForce(self, Rwst, cNormal, cGap, cPointCoordsWheel,
                              cPointCoordsRail, wstPosition, wstVelocity,
                              cElement, cForce, cNormalStiffness=525e6):
        """
        Calculate normal and creep forces.

        Parameters
        ----------
        Rwst : TYPE
            Wheelset rotation matrix.
        cNormal : TYPE
            Contact normal vector (YZ).
        cGap : TYPE
            Size of contact gap.
        cPointCoordsWheel : TYPE
            Contact point position on local contact plane coordinates.
        cPointCoordsRail : TYPE
            Contact point position on local contact plane coordinates.
        wstPosition : array
            Wheelset c.m. position in global coordinates (translation and
                                                          rotation)
        wstVelocity : array
            Wheelset c.m. velocity (translation and rotation)
        cElement : TYPE
            Contacting element.
        cForce : array
            Contact force vector (output)
        cNormalStiffness : double
            Normal contact stiffness. Defaults to 525 MN/m.

        Returns
        -------
        None.

        """
        # normal vector in global coordinates
        normalGlobalCoords, _ = hf.unitaryVector(Rwst.transpose().dot(
            np.insert(cNormal, 0, 0)))
        # 2d contact force on the wheel midplane
        contactForceMag = cNormalStiffness * cGap
        contactForce = contactForceMag * normalGlobalCoords
        # lever arm vector of the contact force on wheelset
        rhoM2star = Rwst.transpose().dot(
            np.insert(cPointCoordsWheel, 0, 0))
        # add force to wheelset
        cForce[self.wheelset.globalDof[:3]] += contactForce
        if contactForce[2] > 0:
            print(
                '\nWarning: right negative contact force {} N'.format(
                    contactForce[2]))
        # add moments to wheelset
        cForce[self.wheelset.globalDof[3:]] += hf.skew(rhoM2star).dot(
            contactForce)
        # convert rail contact point to global coords
        localXi = cElement.mapToLocalCoords(
            wstPosition[:3] + Rwst.transpose().dot(
                np.insert(cPointCoordsRail, 0, 0.0))
        )

        # normal contact force on elements
        rail = cElement.parentBody
        railDofs = np.array(rail.globalDof)
        cForce[railDofs[cElement.globalDof]] += np.dot(
            -contactForce,
            cElement.shapeFunctionMatrix(*localXi)
        )

        # creep forces
        # creepages
        wstOmega = self.wheelset.omega(wstPosition[3:], wstVelocity[3:])
        cPointsVelocityWheel = wstVelocity[:3] + \
            hf.skew(wstOmega).dot(rhoM2star)
        cPointsVelocityRail = cElement.interpolateVelocity(*localXi)

        relativeVelocity = cPointsVelocityWheel - cPointsVelocityRail
        # project into normal
        relativeVelNormal = relativeVelocity.dot(normalGlobalCoords)
        relativeVelocityTang = relativeVelocity - relativeVelNormal
        relativeVelocityTang = Rwst.transpose().dot(relativeVelocityTang)

        creepages = np.array([0., 0.])

        creepages[0] = 0.0 if wstVelocity[0] == 0.0 else (
            relativeVelocityTang[0]) / wstVelocity[0]
        creepages[1] = wstPosition[5]

        fricForce = np.zeros(3)
        # fricForce[0], fricForce[1] = pcf.polach(contactForceMag,
        #                                         0.4, 6e-3, 6e-3, creepages[0],
        #                                         creepages[1], 0.0,
        #                                         4.12, 3.67, 1.47)
        fricForce[0] = self.coulombFric(creepages[0], 0.4) * contactForceMag

        # print('{}:{} N'.format(cElement.parentBody.name, fricForce[0]))
        # apply creep forces to wheel
        if np.any(fricForce) != 0:
            cForce[self.wheelset.globalDof[:3]] += fricForce
            cForce[self.wheelset.globalDof[3:]] += hf.skew(rhoM2star).dot(
                fricForce)

            # creep force on elements
            cForce[railDofs[cElement.globalDof]] += np.dot(
                -fricForce,
                cElement.shapeFunctionMatrix(*localXi))

    def coulombFric(self, v, mu):
        return mu * np.tanh(100*v)

    def evaluateGapFunction(self, t, p, v, leftOrRight='left', plotFlag=False):
        """
        Implememt gap evaluation.

        Parameters
        ----------
        t: double
            Time.
        p: double vector
            System state vector
        v: double vector
            Time derivative of p
        cont: str
            Either 'left' or 'right'

        Returns
        -------
        cNormalLeft: np.array.
            normal contact vector
        cPoints: dictionary
            the contact points written on the wheelset reference frame
        gap: double
            the penetration gap (negative if interference occur)

        """
        wstBody = self.wheelset
        railBody = self.leftRail if leftOrRight == 'left' else self.rightRail

        railProfile = railBody.profiles[0]

        # wheelset reference marker position
        wstPosition = p[wstBody.globalDof[:3]]
        if leftOrRight == 'left':
            wheelProfile = wstBody.getLeftProfile()
        elif leftOrRight == 'right':
            wheelProfile = wstBody.getRightProfile()
        # dofs of rail bodies
        railDof = np.array(railBody.globalDof)

        # gets wheel Cardan angles and removes rotation around shaft axis
        cardans = p[wstBody.globalDof[3:]]
        cardans[1] = 0
        # gets wheelset rotation matrix
        Rwst = hf.cardanRotationMatrix(cardans)

        # now, I've got to find the contacting element
        # we will suppose that, for every contacting element, the terminal
        # nodes must be on opposite sides of the wheelset midplane, i.e.,
        # extreme bending on contacting elements is not allowed
        # Then, we check for each element whether the vector joining its
        # terminal nodes pierces the wheelset midplane.
        midplaneNormal = Rwst[:, 0]
        for e in railBody.elementList:
            p1 = e.nodes[0].qtotal[:3]
            p2 = e.nodes[-1].qtotal[:3]

            # r12hat: unit vector in the direction of the element axis
            # l12: length between terminal nodes of the element
            r12hat, l12 = hf.unitaryVector(p2-p1)

            # calculate gamma_I, which is the length parameter
            gamma_I = np.dot(wstPosition - p1, midplaneNormal) / \
                np.dot(r12hat, midplaneNormal)

            # if gamma_I is positive, but smaller than the distance between
            # terminal nodes, then we've found the contacting element
            if 0 <= gamma_I < l12:
                break

        intersectionP = p1 + gamma_I * r12hat

        # now we know that e is the contacting element
        # it is necessary to find the longitudinal position of the contact plane
        # we perform a very naive bissection search to find it

        # Nelder-Mead algorithm to find the closest point in spine
        xi_init = -1 + 2 * gamma_I / l12
        xi = xi_init
        f = np.linalg.norm(
            intersectionP - e.interpolatePosition(xi_init, 0, 0))
        alpha = 1
        gamma = 2
        rho = 0.5
        tol = 1e-6
        for _ in range(100):
            # Reflection
            xi_r = xi + alpha * (xi - xi_init)
            _, f_r = hf.unitaryVector(intersectionP -
                                      e.interpolatePosition(xi_r, 0, 0))

            if f <= f_r < f + tol:
                xi = xi_r
                f = f_r
            elif f_r < f:
                # Expansion
                xi_e = xi + gamma * (xi_r - xi)
                _, f_e = hf.unitaryVector(intersectionP -
                                          e.interpolatePosition(xi_e, 0, 0))
                if f_e < f_r:
                    xi = xi_e
                    f = f_e
                else:
                    xi = xi_r
                    f = f_r
            else:
                # Contraction
                xi_c = xi + rho * (xi_init - xi)
                _, f_c = hf.unitaryVector(intersectionP -
                                          e.interpolatePosition(xi_c, 0, 0))
                if f_c < f:
                    xi = xi_c
                    f = f_c
                else:
                    # Shrink
                    xi_init = (xi + xi_init) / 2
                    xi = xi_init
                    _, f = hf.unitaryVector(intersectionP -
                                            e.interpolatePosition(xi, 0, 0))

            if np.abs(f - f_r) < tol:
                break

        # compute the projection of rail profile onto wheelset plane
        i = 0
        # railOnWheelsetPlane are the profile points written in
        # global coordinates
        railOnWheelsetPlane = np.zeros([railBody.profiles[0].nPoints, 3])
        profilePoints = e.getProfileGlobalCoordinates(xi)
        for profPoint in profilePoints:
            rhoApG = profPoint - wstPosition -\
                np.dot(profPoint - wstPosition, midplaneNormal) *\
                midplaneNormal
            railOnWheelsetPlane[i] = np.dot(Rwst, rhoApG.T).T
            i = i + 1

        # now auxiliary profiles are created to represent the
        # projected geometries
        railProfile = planarProfile('Rail profile projected')
        if leftOrRight == 'right':
            railProfile.setProfilePoints(
                np.flip(railOnWheelsetPlane[:, 1:3], 0))
        else:
            railProfile.setProfilePoints(railOnWheelsetPlane[:, 1:3])

        # get the convex subsets of the profiles
        wpConvSubsets = (wheelProfile.createConvexSubsets()).copy()
        rpConvSubsets = (railProfile.createConvexSubsets()).copy()

        # find the contact point, if any
        cSubsets = {"rail": None, "wheel": None}
        cPoints = {"rail": None, "wheel": None}
        minDist = np.inf

        for wSubset in wpConvSubsets:
            for rSubset in rpConvSubsets:
                # if rSubset[-1, 0] > wSubset[1, 0]:
                pRail, pWheel, n, d = gjk.gjk(
                    rSubset, wSubset, np.array([0., 1.]))
                # plt.plot(*rSubset.T)
                # plt.plot(*wSubset.T)
                # plt.plot(*pRail, 'x')
                # plt.plot(*pWheel, 'o')
                if d < minDist:
                    minDist = d
                    cSubsets["rail"] = rSubset
                    cSubsets["wheel"] = wSubset
                    cPoints["rail"] = pRail
                    cPoints["wheel"] = pWheel
                    cNormal = n

        if plotFlag:
            y = railOnWheelsetPlane[:, 1]
            z = railOnWheelsetPlane[:, 2]
            plt.plot(y, z, label='Trilho projetado')

            y = wheelProfile.points[:, 0]
            z = wheelProfile.points[:, 1]
            plt.plot(y, z, label='Roda')

            plt.show()
            plt.gca().invert_yaxis()
            plt.legend()

            plt.plot(cPoints['wheel'][0], cPoints['wheel'][1], 'o')
            plt.plot(cPoints['rail'][0], cPoints['rail'][1], 'x')

        if cNormal[1] < 0:
            cNormal = -cNormal

        return cNormal, cPoints, minDist, e
