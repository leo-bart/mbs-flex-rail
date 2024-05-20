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
from profiles import planarProfile


class wrContact2 (MBS.BodyConnections.Contacts.Contact.contact):
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
        wstPosition = p[self.wheelset.globalDof[:3]]
        # gets wheel Cardan angles and removes rotation around shaft axis
        cardans = p[self.wheelset.globalDof[3:]]
        cardans[1] = 0
        # gets wheelset rotation matrix
        Rwst = hf.cardanRotationMatrix(cardans)

        cNormalLeft, cPointsLeft, gapLeft, elemLeft = self.evaluateGapFunction(
            t, p, v, 'left')
        cNormalRight, cPointsRight, gapRight, elemRight = self.evaluateGapFunction(
            t, p, v, 'right')

        f = np.zeros_like(p)
        # left rail
        if gapLeft < 0.0:
            # normal vector in global coordinates
            normalGlobalCoords = Rwst.transpose().dot(np.insert(
                cNormalLeft, 0, 0))
            # 2d contact force on the wheel midplane
            contactForce = 525e6 * gapLeft * normalGlobalCoords
            # lever arm vector of the contact force on wheelset
            rhoM2star = Rwst.transpose().dot(
                np.insert(cPointsLeft['wheel'], 0, 0))
            # add force to wheelset
            f[self.wheelset.globalDof[:3]] += contactForce
            if f[self.wheelset.globalDof[2]] > 0:
                print(
                    '\nWarning: left negative contact force {} N'.format(
                        f[self.wheelset.globalDof[2]]))
            f[self.wheelset.globalDof[3:]
              ] += hf.skew(rhoM2star).dot(contactForce)
            # convert rail contact point to global coords
            localXi = elemLeft.mapToLocalCoords(
                wstPosition + Rwst.transpose().dot(
                    np.insert(cPointsLeft["rail"], 0, 0.0))
            )

            # normal contact force
            railDofs = np.array(self.leftRail.globalDof)
            f[railDofs[elemLeft.globalDof]] += np.dot(
                -contactForce,
                elemLeft.shapeFunctionMatrix(*localXi)
            )
        # right rail
        if gapRight < 0.0:
            # normal vector in global coordinates
            normalGlobalCoords = Rwst.transpose().dot(
                np.insert(cNormalRight, 0, 0))
            # 2d contact force on the wheel midplane
            contactForce = 525e6 * gapRight * normalGlobalCoords
            # lever arm vector of the contact force on wheelset
            rhoM2star = Rwst.transpose().dot(
                np.insert(cPointsRight['wheel'], 0, 0))
            # add force to wheelset
            f[self.wheelset.globalDof[:3]] += contactForce
            if f[self.wheelset.globalDof[2]] > 0:
                print(
                    '\nWarning: right negative contact force {} N'.format(
                        f[self.wheelset.globalDof[2]]))
            f[self.wheelset.globalDof[3:]
              ] += hf.skew(rhoM2star).dot(contactForce)
            # convert rail contact point to global coords
            localXi = elemRight.mapToLocalCoords(
                wstPosition + Rwst.transpose().dot(
                    np.insert(cPointsRight["rail"], 0, 0.0))
            )

            # normal contact force
            railDofs = np.array(self.rightRail.globalDof)
            f[railDofs[elemRight.globalDof]] += np.dot(
                -contactForce,
                elemRight.shapeFunctionMatrix(*localXi)
            )

        return f

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

        # Nelder-Mead algorithm to find the closes point in spine
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


# %%


class wrContact (MBS.BodyConnections.Contacts.Contact.contact):
    """
    Wheel-rail contact force.

    Defines a wheel-rail contact object.

    """

    def __init__(self, name_='Contact force'):
        super().__init__(name_)

    def connect(self, _masterProfile, _slaveProfile):
        self.masterProfile = _masterProfile
        self.slaveProfile = _slaveProfile

        super().connect(_masterProfile.parent, _slaveProfile.parent)

    def evaluateForceFunction(self, t, p, v, *args):
        """
        Caculate wheel-rail contact force.

        The wheel profile coordinates is referenced on the wheelset center, i.e.,
        on the coordinate system placed at the center of the shaft.

        Parameters
        ----------
        t : array
            Time t.
        p : array
            position at time t.
        v : array
            velocity at time t.
        m1 : marker
            rail profile reference marker.
        m2 : marker
            wheel profile reference marker.

        Returns
        -------
        f : array
            force.

        """
        # gets rail and wheelset bodies
        railBody = self.masterProfile.parent
        wstBody = self.slaveProfile.parent

        railReferenceMarker = self.masterProfile.referenceMarker
        wheelReferenceMarker = self.slaveProfile.referenceMarker
        # wheelset reference marker position
        wstP = p[wstBody.globalDof[:3]]
        # dofs of rail body
        railDof = np.array(railBody.globalDof)

        # gets wheel Cardan angles and removes rotation around shaft axis
        cardans = p[wstBody.globalDof[3:]]
        cardans[2] = 0
        # gets wheelset rotation matrix
        Rwst = hf.cardanRotationMatrix(cardans)
        # rhoM2 is the relative position of the wheel profile, represented on
        # the glogal reference frame
        rhoM2 = Rwst.dot(wheelReferenceMarker.position)
        # pWheel is the position of the wheel profile reference point on
        # the global reference frame
        pWheel = wstP + rhoM2

        # matrix to convert vectors written on the wheelset reference frame
        # to the profile reference frame
        """TODO: eu estou usando aqui o sistema de referência do perfil
        como se a sua orientação fosse só girar o sistema do rodeiro, mas 
        não é só isso.
        
        Ideia para o momento: verificar resto da rotina para ver 
        se é necessário mesmo usar esse sistema ou se dá para fazer as
        conversões diretamente usando o referenceMarker.orientation 
        do perfil.
        """
        wst2prof = np.array([[0, 0, 1], [0, 1, 0]])

        # wheelset reference frame position on profile reference frame coordinates
        wstPp = wst2prof.dot(wstP)

        # profiles
        wp = self.slaveProfile
        rp = self.masterProfile

        # now, I've got to find the contacting element
        # we will suppose that, for every contacting element, the terminal
        # nodes must be on opposite sides of the wheelset midplane, i.e.,
        # extreme bending on contacting elements is not allowed
        # Then, we check for each element whether the vector joining its
        # terminal nodes pierces the wheelset midplane.
        midplaneNormal = Rwst[:, 0]
        for e in railBody.elementList:
            n1 = e.nodes[0]
            n2 = e.nodes[-1]

            # projects the distance between the front node and end
            # node of each element e to the wheel reference point
            d1 = (n1.qtotal[:3] - pWheel).dot(midplaneNormal)
            d2 = (n2.qtotal[:3] - pWheel).dot(midplaneNormal)

            # if the signs of d1 and d2 are different, than the element
            # pierces the midplane and the element e is the contacting element
            # the loop can be broken
            if d1*d2 <= 0:
                break

        # now e is the contact element
        # it is necessary to find the longitudinal position of the contact plane
        # we perform a very naive bissection search to find it

        # start with finding the node that is closer to the plane
        # direction tells the direction of the first search bissection
        dmin = d1
        step = 2
        startXi = -1
        newXi = startXi
        while dmin*dmin > 1e-7:
            # in the following loop, the `newXi` variable outputs the approximate
            # xi coordinate of the contact point
            newd = -(pWheel - e.interpolatePosition(newXi+step, 0, 0)
                     ).dot(midplaneNormal)
            while newd*dmin > 0:
                step *= 1.2
                newXi = newXi+step
                newd = -(pWheel - e.interpolatePosition(newXi +
                         step, 0, 0)).dot(midplaneNormal)
            dmin = newd
            newXi += step
            step = -step/2

        railCpointPosi = e.interpolatePosition(newXi, 1, 0)  # note eta = 1

        '''
        O que eu preciso fazer aqui?
        
        A variável railCpointPosi determina a posição global do ponto de
        interseção do trilho com o plano médio vertical do rodeiro.
        
        Preciso:
            1. obter a orientação da seção transversal neste ponto. Não desen-
            volvi uma função específica para interpolar orientação, então vou
            ter que fazer isso aqui na rotina mesmo. Talvez seja bom, no futuro,
            passar para dentro do elemento para aproveitar o Cython. Quando
            fizer isso, é necessário atentar-se para o fato de a orientação
            do perfil não ser necessariamente a mesma do trilho
            2. transferir o marker de referência do perfil do trilho para o 
            railCpointPosi com a orientação obtida.
        '''

        ##########
        # we can now search for the contact point between wheel and rail profiles
        ##########
        plot = False  # set to TRUE to plot wheel and rail profiles
        if plot:
            x = rp.points[:, 0] + railCpointPosi[2]
            y = rp.points[:, 1] + railCpointPosi[1]
            plt.plot(x, y)

            x = wp.points[:, 0] + wstP[2]
            y = wp.points[:, 1] + wstP[1]
            plt.plot(x, y)

        # we get the convex subsets of the wheel and rail profiles and
        # offset them to the global position

        if pWheel[2] < 0:
            wstFactor = -1
        else:
            wstFactor = 1

        wp = wstBody.profiles[0]
        # we make a copy to preserve the original profile
        wpConvSubsets = (wp.createConvexSubsets()).copy()
        # rotation matrix of the wheelset on the profile css
        A = wst2prof.dot(Rwst.dot(wst2prof.transpose()))
        for i in range(len(wpConvSubsets)):
            wpConvSubsets[i][:, 0] += wstPp[0]
            wpConvSubsets[i][:, 0] *= wstFactor
            wpConvSubsets[i][:, 1] += wstPp[1]
            wpConvSubsets[i] = wpConvSubsets[i].dot(A)
            if plot:
                plt.plot(wpConvSubsets[i][:, 0], wpConvSubsets[i][:, 1])

        rp = railBody.profiles[0]
        # rpConvSubsets = (rp.createConvexSubsets()).copy()
        # headOffset = 0.01 # offset to artificially increase head height
        #                   # this prevent degenerate contact conditions when
        #                   # wheel penetration is large compared to convex subset
        #                   # total height
        # for i in range(len(rpConvSubsets)):
        #     rpConvSubsets[i][:,0] += railCpointPosi[2]
        #     rpConvSubsets[i][:,1] += railCpointPosi[1]
        #     rpConvSubsets[i] = np.append(rpConvSubsets[i],[rpConvSubsets[i][-1,:]],axis=0)
        #     rpConvSubsets[i] = np.append(rpConvSubsets[i],[rpConvSubsets[i][0,:]],axis=0)
        #     rpConvSubsets[i][-1,1] -= headOffset
        #     rpConvSubsets[i][-2,1] -= headOffset

        #     if plot:
        #         plt.plot(rpConvSubsets[i][:,0],rpConvSubsets[i][:,1])

        # replace all convex subsets by the original rail profile
        # this is to try and converge
        rpConvSubsets = []
        rpConvSubsets.append(rp.points.copy())
        rpConvSubsets[0][:, 0] += railCpointPosi[2]
        rpConvSubsets[0][:, 1] += railCpointPosi[1]

        # find the contact point, if any
        cSubsets = {"rail": None, "wheel": None}
        cPoints = {"rail": None, "wheel": None}
        minDist = np.inf

        for rSubset in rpConvSubsets:
            for wSubset in wpConvSubsets:
                if rSubset[-1, 0] > wSubset[1, 0]:
                    pRail, pWheel, n, d = gjk.gjk(
                        rSubset, wSubset, np.array([0., -1.]))
                    # print(d)
                    if d < minDist:
                        minDist = d
                        cSubsets["rail"] = rSubset
                        cSubsets["wheel"] = wSubset
                        cPoints["rail"] = pRail
                        cPoints["wheel"] = pWheel
                        cNormal = n

        # plt.fill(cSubsets["rail"][:,0],cSubsets["rail"][:,1], edgecolor='blue')
        # plt.fill(cSubsets["wheel"][:,0],cSubsets["wheel"][:,1], edgecolor='orange')
        # print(minDist)

        f = np.zeros_like(p)
        if minDist < 0.0:
            # 2d contact force on the wheel midplane
            contactForce = 525e6 * minDist * wst2prof.transpose().dot(cNormal)
            # gets the vector from the wheelset CoG to the contact point
            # first on profile local coordinates
            rhoM2star = cPoints["wheel"] - wstPp
            # then on global coordinates
            rhoM2star = Rwst.transpose().dot(wst2prof.transpose().dot(rhoM2star))

            f[wstBody.globalDof[:3]] += contactForce
            if f[-5] < 0:
                print('Warning: negative contact force {} N'.format(f[-5]))
            f[wstBody.globalDof[3:]] += hf.skew(rhoM2star).dot(contactForce)

            cPoints["rail"] = Rwst.transpose().dot(
                wst2prof.transpose().dot(cPoints["rail"]))
            localXi = e.mapToLocalCoords(cPoints["rail"])

            # normal contact force
            f[railDof[e.globalDof]] -= np.dot(
                contactForce,
                e.shapeFunctionMatrix(localXi[0], localXi[1], localXi[2])
            )

            # tangential contact force
            # encontrar velocidade dos pontos de contato
            # aplicar modelo de Kalker

        return f

    def evaluateGapFunction(self, t, p, v, *args):
        # gets rail and wheelset bodies
        railBody = self.masterProfile.parent
        wstBody = self.slaveProfile.parent

        railReferenceMarker = self.masterProfile.referenceMarker
        wheelReferenceMarker = self.slaveProfile.referenceMarker
        # wheelset reference marker position
        wstP = p[wstBody.globalDof[:3]]
        # dofs of rail body
        railDof = np.array(railBody.globalDof)

        # gets wheel Cardan angles and removes rotation around shaft axis
        cardans = p[wstBody.globalDof[3:]]
        cardans[2] = 0
        # gets wheelset rotation matrix
        Rwst = hf.cardanRotationMatrix(cardans)
        # rhoM2 is the relative position of the wheel profile, represented on
        # the glogal reference frame
        rhoM2 = Rwst.dot(wheelReferenceMarker.position)
        # pWheel is the position of the wheel profile reference point on
        # the global reference frame
        pWheel = wstP + rhoM2

        # matrix to convert vectors written on the wheelset reference frame
        # to the profile reference frame
        """TODO: eu estou usando aqui o sistema de referência do perfil
        como se a sua orientação fosse só girar o sistema do rodeiro, mas 
        não é só isso.
        
        Ideia para o momento: verificar resto da rotina para ver 
        se é necessário mesmo usar esse sistema ou se dá para fazer as
        conversões diretamente usando o referenceMarker.orientation 
        do perfil.
        """
        wst2prof = np.array([[0, 0, 1], [0, 1, 0]])

        # wheelset reference frame position on profile reference frame coordinates
        wstPp = wst2prof.dot(wstP)

        # profiles
        wp = self.slaveProfile
        rp = self.masterProfile

        # now, I've got to find the contacting element
        # we will suppose that, for every contacting element, the terminal
        # nodes must be on opposite sides of the wheelset midplane, i.e.,
        # extreme bending on contacting elements is not allowed
        # Then, we check for each element whether the vector joining its
        # terminal nodes pierces the wheelset midplane.
        midplaneNormal = Rwst[:, 0]
        for e in railBody.elementList:
            n1 = e.nodes[0]
            n2 = e.nodes[-1]

            # projects the distance between the front node and end
            # node of each element e to the wheel reference point
            d1 = (n1.qtotal[:3] - pWheel).dot(midplaneNormal)
            d2 = (n2.qtotal[:3] - pWheel).dot(midplaneNormal)

            # if the signs of d1 and d2 are different, than the element
            # pierces the midplane and the element e is the contacting element
            # the loop can be broken
            if d1*d2 <= 0:
                break

        # now e is the contact element
        # it is necessary to find the longitudinal position of the contact plane
        # we perform a very naive bissection search to find it

        # start with finding the node that is closer to the plane
        # direction tells the direction of the first search bissection
        dmin = d1
        step = 2
        startXi = -1
        newXi = startXi
        while dmin*dmin > 1e-7:
            # in the following loop, the `newXi` variable outputs the approximate
            # xi coordinate of the contact point
            newd = -(pWheel - e.interpolatePosition(newXi+step, 0, 0)
                     ).dot(midplaneNormal)
            while newd*dmin > 0:
                step *= 1.2
                newXi = newXi+step
                newd = -(pWheel - e.interpolatePosition(newXi +
                         step, 0, 0)).dot(midplaneNormal)
            dmin = newd
            newXi += step
            step = -step/2

        railCpointPosi = e.interpolatePosition(newXi, 1, 0)  # note eta = 1

        '''
        O que eu preciso fazer aqui?
        
        A variável railCpointPosi determina a posição global do ponto de
        interseção do trilho com o plano médio vertical do rodeiro.
        
        Preciso:
            1. obter a orientação da seção transversal neste ponto. Não desen-
            volvi uma função específica para interpolar orientação, então vou
            ter que fazer isso aqui na rotina mesmo. Talvez seja bom, no futuro,
            passar para dentro do elemento para aproveitar o Cython. Quando
            fizer isso, é necessário atentar-se para o fato de a orientação
            do perfil não ser necessariamente a mesma do trilho
            2. transferir o marker de referência do perfil do trilho para o 
            railCpointPosi com a orientação obtida.
        '''

        ##########
        # we can now search for the contact point between wheel and rail profiles
        ##########
        plot = False  # set to TRUE to plot wheel and rail profiles
        if plot:
            x = rp.points[:, 0] + railCpointPosi[2]
            y = rp.points[:, 1] + railCpointPosi[1]
            plt.plot(x, y)

            x = wp.points[:, 0] + wstP[2]
            y = wp.points[:, 1] + wstP[1]
            plt.plot(x, y)

        # we get the convex subsets of the wheel and rail profiles and
        # offset them to the global position

        if pWheel[2] < 0:
            wstFactor = -1
        else:
            wstFactor = 1

        wp = wstBody.profiles[0]
        # we make a copy to preserve the original profile
        wpConvSubsets = (wp.createConvexSubsets()).copy()
        # rotation matrix of the wheelset on the profile css
        A = wst2prof.dot(Rwst.dot(wst2prof.transpose()))
        for i in range(len(wpConvSubsets)):
            wpConvSubsets[i][:, 0] += wstPp[0]
            wpConvSubsets[i][:, 0] *= wstFactor
            wpConvSubsets[i][:, 1] += wstPp[1]
            wpConvSubsets[i] = wpConvSubsets[i].dot(A)
            if plot:
                plt.plot(wpConvSubsets[i][:, 0], wpConvSubsets[i][:, 1])

        rp = railBody.profiles[0]
        # rpConvSubsets = (rp.createConvexSubsets()).copy()
        # headOffset = 0.01 # offset to artificially increase head height
        #                   # this prevent degenerate contact conditions when
        #                   # wheel penetration is large compared to convex subset
        #                   # total height
        # for i in range(len(rpConvSubsets)):
        #     rpConvSubsets[i][:,0] += railCpointPosi[2]
        #     rpConvSubsets[i][:,1] += railCpointPosi[1]
        #     rpConvSubsets[i] = np.append(rpConvSubsets[i],[rpConvSubsets[i][-1,:]],axis=0)
        #     rpConvSubsets[i] = np.append(rpConvSubsets[i],[rpConvSubsets[i][0,:]],axis=0)
        #     rpConvSubsets[i][-1,1] -= headOffset
        #     rpConvSubsets[i][-2,1] -= headOffset

        #     if plot:
        #         plt.plot(rpConvSubsets[i][:,0],rpConvSubsets[i][:,1])

        # replace all convex subsets by the original rail profile
        # this is to try and converge
        rpConvSubsets = []
        rpConvSubsets.append(rp.points.copy())
        rpConvSubsets[0][:, 0] += railCpointPosi[2]
        rpConvSubsets[0][:, 1] += railCpointPosi[1]

        # find the contact point, if any
        cSubsets = {"rail": None, "wheel": None}
        cPoints = {"rail": None, "wheel": None}
        minDist = np.inf

        for rSubset in rpConvSubsets:
            for wSubset in wpConvSubsets:
                if rSubset[-1, 0] > wSubset[1, 0]:
                    pRail, pWheel, n, d = gjk.gjk(
                        rSubset, wSubset, np.array([0., -1.]))
                    # print(d)
                    if d < minDist:
                        minDist = d
                        cSubsets["rail"] = rSubset
                        cSubsets["wheel"] = wSubset
                        cPoints["rail"] = pRail
                        cPoints["wheel"] = pWheel
                        cNormal = n

        # plt.fill(cSubsets["rail"][:,0],cSubsets["rail"][:,1], edgecolor='blue')
        # plt.fill(cSubsets["wheel"][:,0],cSubsets["wheel"][:,1], edgecolor='orange')
        # print(minDist)

        return d, d

    def plotContactPosition(self, t, p, v):
        pass
