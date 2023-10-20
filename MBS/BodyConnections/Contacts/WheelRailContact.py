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

class wrContact (MBS.BodyConnections.Contacts.Contact.contact):
    """
    Wheel-rail contact force.
    
    Defines a wheel-rail contact object.
    
    """
    
    def __init__(self,name_='Contact force'):
        super().__init__(name_)
        
    def connect(self,_masterProfile,_slaveProfile):
        self.masterProfile = _masterProfile
        self.slaveProfile = _slaveProfile
        
        super().connect(_masterProfile.parent,_slaveProfile.parent)
        
    def evaluateForceFunction(self,t,p,v,*args):
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
        m1 = args[0]
        m2 = args[1]
        
        # gets rail and wheelset bodies
        railBody = m1.parent
        wstBody = m2.parent
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
        rhoM2 = Rwst.dot(m2.position)
        # pWheel is the position of the wheel profile reference point on
        # the global reference frame
        pWheel = wstP + rhoM2
        
        # matrix to convert vectors written on the wheelset reference frame
        # to the profile reference frame
        wst2prof = np.array([[0,0,1],[0,1,0]])
        
        # wheelset reference frame position on profile reference frame coordinates
        wstPp = wst2prof.dot(wstP)
        
        
        
        # profiles
        wp = wstBody.profiles[0]
        rp = railBody.profiles[0]
        
        # now, I've got to find the contacting element
        # we will suppose that, for every contacting element, the terminal
        # nodes must be on opposite sides of the wheelset midplane, i.e.,
        # extreme bending on contacting elements is not allowed
        # Then, we check for each element whether the vector joining its
        # terminal nodes pierces the wheelset midplane.
        midplaneNormal = Rwst[:,0]
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
            newd = -(pWheel - e.interpolatePosition(newXi+step,0,0)).dot(midplaneNormal)
            while newd*dmin > 0:
                step *= 1.2
                newXi = newXi+step
                newd = -(pWheel - e.interpolatePosition(newXi+step,0,0)).dot(midplaneNormal)
            dmin = newd
            newXi +=step
            step = -step/2
        
        railCpointPosi = e.interpolatePosition(newXi,1,0) # note eta = 1
         
        ########## 
        # we can now search for the contact point between wheel and rail profiles
        ##########
        plot = False # set to TRUE to plot wheel and rail profiles
        if plot:
            x = rp.points[:,0] + railCpointPosi[2]
            y = rp.points[:,1] + railCpointPosi[1]
            plt.plot(x,y)
            
            x = wp.points[:,0] + wstP[2]
            y = wp.points[:,1] + wstP[1]
            plt.plot(x,y)
        
        
        # we get the convex subsets of the wheel and rail profiles and
        # offset them to the global position
        
        if pWheel[2] < 0:
            wstFactor = -1
        else:
            wstFactor = 1
        
        wp = wstBody.profiles[0]
        wpConvSubsets = (wp.createConvexSubsets()).copy() # we make a copy to preserve the original profile
        A=wst2prof.dot(Rwst.dot(wst2prof.transpose())) # rotation matrix of the wheelset on the profile css
        for i in range(len(wpConvSubsets)):
            wpConvSubsets[i][:,0] += wstPp[0]
            wpConvSubsets[i][:,0] *= wstFactor
            wpConvSubsets[i][:,1] += wstPp[1]
            wpConvSubsets[i] = wpConvSubsets[i].dot(A)
            if plot:
                plt.plot(wpConvSubsets[i][:,0],wpConvSubsets[i][:,1])
            
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
        rpConvSubsets[0][:,0] += railCpointPosi[2]
        rpConvSubsets[0][:,1] += railCpointPosi[1]
        
        
        # find the contact point, if any
        cSubsets = {"rail":None,"wheel":None}
        cPoints = {"rail":None,"wheel":None}
        minDist = np.inf
        
        for rSubset in rpConvSubsets:
            for wSubset in wpConvSubsets:
                if rSubset[-1,0] > wSubset[1,0]:
                    pRail,pWheel,n,d = gjk.gjk(rSubset,wSubset,np.array([0.,-1.]))
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
            
            cPoints["rail"] = Rwst.transpose().dot(wst2prof.transpose().dot(cPoints["rail"]))
            localXi = e.mapToLocalCoords(cPoints["rail"])
            
            # normal contact force
            f[railDof[e.globalDof]] -=  np.dot(
                contactForce, 
                e.shapeFunctionMatrix(localXi[0],localXi[1],localXi[2])
                )
            
            # tangential contact force
            # encontrar velocidade dos pontos de contato
            # aplicar modelo de Kalker
            

            
        return f
    
    
    def plotContactPosition(self,t,p,v):
        pass
        
