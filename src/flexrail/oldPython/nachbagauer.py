#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the planar beam in 
K. Nachbagauer, A. Pechstein, H. Irschik, e J. Gerstmayr, “A new locking-free 
formulation for planar, shear deformable, linear and quadratic 
beam finite elements based on the absolute nodal coordinate formulation”, 
Multibody System Dynamics, vol. 26, no 3, p. 245–263, 2011,
doi: 10.1007/s11044-011-9249-8.


Created on Thu May 13 14:49:04 2021

@author: Leonardo Bartalini Baruffaldi
"""

import numpy as np

from numpy.matlib import matrix, eye
from numpy import dot
from numpy.linalg import norm, inv



#import flexibleBody
#import materials


class node(object):
    """
    finite element node with four dof
    
    Parameters
    __________
        u1 - reference coordinate x
        u2 - reference coordinate y
        up1 - reference slope relative to x
        up2 - reference slope relative to y
    """
    
    def __init__(self, u1=0., u2=0., up1 = 0., up2 = 1.):
        self.q0 = [u1,u2,up1,up2]
        self.q = [0.0]*4
        self.globalDof = [0,1,2,3]
        
            
        
    @property
    def q(self):
        """nodal displacement relative to reference configuration"""
        return self._q
    @q.setter
    def q(self,dofMatrix):
        self._q = dofMatrix
        #self.qtotal = np.array(dofMatrix) + np.array(self.q0)
        
    def updateqTotal(self):
        self.qtotal = np.array(self.q, dtype=np.float64) + np.array(self.q0, dtype=np.float64)
        
    
    @property
    def qtotal(self):
        """nodal displacement relative to global frame"""
        return np.array(self.q) + np.array(self.q0)


    
        
        
        
        
        
         

    
########################################
class beamANCFelement(object):
    '''
    Base class for finite elements
    '''
    def __init__(self):
        self.parentBody = None
        self.length = 1.0
        self.height = 1.0
        self.width = 1.0
        self.nodalElasticForces = None
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
        self.nodes = []
        
        
       
    
    @property
    def q0(self):
        q0 = []
        for nd in self.nodes:
            q0 = q0 + nd.q0
        return q0
    
    @property
    def gaussIntegrationPoints(self):
        return {1:[0],
                2:[-1.0/np.sqrt(3),1.0/np.sqrt(3)],
                3:[-0.7745967, 0, 0.7745967],
                4:[-0.86113631,-0.33998104,0.33998104,0.86113631]}
    
    @property
    def gaussWeights(self): 
        return {1:[2],
                2:[1,1],
                3:[0.555556, 0.888889, 0.555556],
                4:[0.33998104,0.6521451549,0.6521451549,0.33998104]}
    
    @property
    def qtotal(self):
        """nodal position relative to global frame"""
        myq = []
        for node in self.nodes:
            myq.extend(node.qtotal.tolist())  
        return myq
    
    @property
    def q(self):
        '''nodal displacement relative to global frame'''
        myq = []
        for node in self.nodes:
            myq.extend(node.q)
            
        return myq
    
    @property
    def globalDof(self):
        gd = []
        for nd in self.nodes:
            gd.extend(nd.globalDof)
        return gd
    
    @property
    def mass(self):
        return self.length * self.height * self.width * self.parentBody.material.rho
    
    def interpolatePosition(self,xi_, eta_,zeta_):
        """
        Returns the interpolated position given the non-dimensional parameters
        xi_ and eta_. Notice that xi_ and eta_ range from -1 to 1.
        """
        
        r = dot(self.shapeFunctionMatrix(xi_ ,eta_), self.qtotal)
        
        return r.reshape(1,-1)
    
    
    def getJacobian(self,xi_,eta_,q=None):
        '''
        

        Parameters
        ----------
        xi_ : DOUBLE
            1st ELEMENT INSTRINSIC COORDINATE [-1,1].
        eta_ : DOUBLE
            2nd ELEMENT INSTRINSIC COORDINATE [-1,1].
        q : VECTOR DOUBLE
            NODAL COORDINATES.

        Returns
        -------
        BLOCK MATRIX
            JACOBIAN CALCULATED AT xi_, eta_, under coordinates q.

        '''
        
        if q == None:
            q = self.qtotal
        nq = len(q)
  
        # TODO remove in future
        # dSx_dxi, dSx_deta, dSy_dxi, dSy_deta = self.shapeFunctionDerivative(xi_,eta_)
        dS = self.shapeFunctionDerivative(xi_, eta_)
        dS = dS.reshape(2,-1)
        
        M1 = dot([dS[:,:nq],dS[:,nq:2*nq]],q).T.round(16)

        return np.asmatrix(M1)
    
    def saveInitialJacobian(self):
        
        jaco = []       
        
        jaco.append([self.initialJacobian(-1,1),self.initialJacobian(1,1)])
        jaco.append([self.initialJacobian(-1,-1),self.initialJacobian(1,-1)])
        
        invJaco = np.linalg.inv(jaco)
        
        detJaco = np.linalg.det(jaco)
        
        if jaco[0][0].all() == jaco[0][1].all() and jaco[0][1].all() == jaco[1][0].all():
            constant = True
        
        return jaco, invJaco, detJaco, constant
        
    
    def loadInitialJacobian(self,xi_,eta_):
        
        if self.isJ0constant:
            J = self.J0[0][0]
            detJ = self.detJ0[0][0]
            invJ = self.invJ0[0][0]
            
        return J,detJ,invJ

    
    def initialJacobian(self,xi_,eta_):
        return self.getJacobian(xi_,eta_,self.q0)
    
    def inverseInitialJacobian(self,xi_,eta_):
        J0 = self.initialJacobian(xi_,eta_)
        return inv(J0)
    
    def currentJacobian(self,xi_,eta_):
        return self.getJacobian(xi_,eta_,self.qtotal)
    
    def getMassMatrix(self):
        
        # Gauss integration points
        gauss = self.gaussIntegrationPoints[3]
        npoints = len(gauss)
        
        # Gauss weights
        w = self.gaussWeights[3]
        
        M = 0*eye(len(self.q),dtype=np.float64)
        
        for i in range(npoints):
            for j in range(npoints):
                S = self.shapeFunctionMatrix(gauss[i],gauss[j])
                M = M + S.T*S * w[i] * w[j]
                
        """we have to multiply by the length and height because
        calculations are carried out on non-dimensional coordinates [-1,1]
        """        
        return self.parentBody.material.rho * M * self.length * self.height * self.width / 4
    
    def getTangentStiffnessMatrix(self):
        '''
        Finite difference approximation to the tangent stiffness matrix

        Returns
        -------
        Kte : Numpy matrix
            Tangent stiffness matrix.

        '''
        
        ndof = len(self.globalDof)
        
        Kte = np.zeros([ndof,ndof])
        
        col = 0
        
        Q0 = self.getNodalElasticForces()
        
        for nd in self.nodes:
            for i,curDof in enumerate(nd.q):
                savePos = curDof
                nd.q[i] += 1e-6
                Kte[:,col] = (self.getNodalElasticForces() - Q0) * 1e6
                nd.q[i] = savePos
                col += 1
                
        return Kte
                
    
    
    def stressTensorByPosition(self,xi_,eta_,split=True):
        return self.parentBody.material.stressTensor(self.strainTensor(xi_, eta_),split)
    
    def strainTensor(self,xi_,eta_,q=None):
        '''
        strainTensor calculates the strain tensor at element coordinates
        (xi_,eta_), both defined between -1 and 1

        Parameters
        ----------
        xi_ : double
            length coordinate.
        eta_ : double
            height coordinate.
        q : array like
            absolute coordinates of nodes

        Returns
        -------
        numpy matrix
            Cauchy-Green strain tensor.

        '''
        if q == None:
            q = self.qtotal
        
        F = self.getJacobian(xi_,eta_,q) * self.loadInitialJacobian(xi_,eta_)[2]
        
        return 0.5 * (dot(F.T,F) - np.matrix([[1.,0.],[0.,1.]]))
    
    
    def shapeFunctionDerivative(self,xi_,eta_):
        raise NotImplementedError()
    
    
    
    def strainTensorDerivative(self,xi_,eta_,q=None):
        '''
        Gets the strain tensor derivative at a certain point of the element
        (given by xi_ and eta_).

        Parameters
        ----------
        xi_ : DOUBLE
            LONGITUDINAL POSITION [-1,1].
        eta_ : DOUBLE
            TRANVERSAL POSITION [-1,1].

        Returns
        -------
        deps_dq : NUMPY.NDARRAY
            3 rd order tensor of the strain tensor derivative.
            deps_dq[:,:,n] can be used to access the n-th slice of the tensor

        '''
        invJ0 = self.loadInitialJacobian(xi_, eta_)[2]
        
        if q == None:
            q = self.qtotal
        
        dSx_dxi, dSx_deta, dSy_dxi, dSy_deta = self.shapeFunctionDerivative(xi_,eta_)
        
        
        # TODO: separate into threads
        U11 = np.sum([np.outer(dSx_dxi,dSx_dxi), np.outer(dSy_dxi,dSy_dxi)],axis=0)
        U12 = np.sum([np.outer(dSx_dxi,dSx_deta), np.outer(dSy_dxi,dSy_deta)], axis=0)
        U21 = U12.T
        U22 = np.sum([np.outer(dSx_deta,dSx_deta), np.outer(dSy_deta,dSy_deta)],axis=0)
            
        qU12,qU11,qU22 = dot(q,[U12+U21,2*U11,2*U22])     
        
        ndof = len(q)
        deps_dq = np.zeros((2,2,ndof),dtype=np.float64)
             
        Ucaped = np.zeros([2,2],dtype=np.float64)

        for m in range(ndof):
            Ucaped[0,0] = qU11[m]
            Ucaped[0,1] = qU12[m]
            Ucaped[1,0] = qU12[m]
            Ucaped[1,1] = qU22[m]
            deps_dq[:,:,m] = 0.5 * dot(dot(invJ0.T,Ucaped),invJ0)

        
       
        return deps_dq 
    
    
    def getNodalElasticForces(self,q = None):
        
        
        # beam geometry
        L = self.length
        H = self.height
        W = self.width
        
        # TODO correct changedStates calculation
   
        if q == None:
            q = self.qtotal
        
        # Gauss integration points
        gaussL = self.gaussIntegrationPoints[2]
        gaussH = self.gaussIntegrationPoints[2]
        
        # Gauss weights
        wL = self.gaussWeights[2]
        wH = self.gaussWeights[2]
        
        
        ndof = len(self.q)
        Qe = np.asarray([0.]*ndof,dtype=np.float64)                                
        # selective reduced integration
        for p in range(len(gaussL)):
            'length quadrature'
            for b in range(len(gaussH)):
                'heigth quadrature'
                detJ0 = self.loadInitialJacobian(gaussL[p], gaussH[b])[1]
                deps_dq = self.strainTensorDerivative(gaussL[p], gaussH[b],q)
                T, Tc = self.parentBody.material.stressTensor(self.strainTensor(gaussL[p], gaussH[b],q),
                                                              split=True)
                Tweight = T * detJ0 * wL[p] * wH[b]
                
                Qe += np.einsum('ij...,ij',deps_dq,Tweight)
                    
                
            # end of height quadrature
            detJ0 = self.loadInitialJacobian(gaussL[p], 0)[1]
            deps_dq = self.strainTensorDerivative(gaussL[p], 0, q)
            T, Tc = self.parentBody.material.stressTensor(self.strainTensor(gaussL[p], 0, q),split=True)
            TcWeight = Tc * detJ0 * wL[p]
            for m in range(ndof):
                Qe[m] += np.multiply(deps_dq[:,:,m],TcWeight).sum()
            #Qe += np.einsum('ij...,ij',deps_dq,Tweight)  
        # end of integration
            
        Qe = Qe * W * L * H / 4
        
        self.nodalElasticForces = Qe
   

        return Qe
    
    
    
    def strainEnergyNorm(self):
        
        q = self.q
        
        # beam geometry
        L = self.length
        H = self.height
        W = self.width
        
        
        # Gauss integration points
        gaussL = self.gaussIntegrationPoints[3]
        gaussH = self.gaussIntegrationPoints[3]
        
        # Gauss weights
        wL = self.gaussWeights[3]
        wH = self.gaussWeights[3]
        
        
        ndof = len(self.q)
        U = 0                               
        # selective reduced integration
        for p in range(len(gaussL)):
            'length quadrature'
            for b in range(len(gaussH)):
                'heigth quadrature'
                detJ0 = self.loadInitialJacobian(gaussL[p], gaussH[b])[1]
                eps = self.strainTensor(gaussL[p], gaussH[b],q)
                T, Tc = self.parentBody.material.stressTensor(eps, split=True)
                Tweight = T * detJ0 * wL[p] * wH[b]
                
                #ts = time()
                U += abs(np.einsum('ij,ij',eps,Tweight))
                #print('{0:1.8g}'.format(time()-ts))
                    
                
            # end of height quadrature
            detJ0 = self.loadInitialJacobian(gaussL[p], 0)[1]
            eps = self.strainTensor(gaussL[p], 0, q)
            T, Tc = self.parentBody.material.stressTensor(eps,split=True)
            TcWeight = Tc * detJ0 * wL[p]
            for m in range(ndof):
                U += abs(np.multiply(eps,TcWeight).sum()) 
        # end of integration
            
        U *= W * L * H / 8
        
        return U / 2
    
    



#%%
class elementLinear(beamANCFelement):
    """
    Planar finite element with linear interpolation
    """
     
    def __init__(self, node1, node2, _height, _width):
        self.length = norm(node1.qtotal[0:2] - node2.qtotal[0:2])
        self.height = _height
        self.width = _width
        self.nodes = [node1,node2]
        self.J0, self.invJ0, self.detJ0, self.isJ0constant = self.saveInitialJacobian()
        self.nodalElasticForces = np.zeros(8,dtype=np.float64)
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
    

    def shapeFunctionMatrix(self, xi_, eta_):

        L = self.length
        xi = xi_ * L/2
        eta = eta_ * self.height / 2

        S1 = (L/2 - xi)
        S2 = eta * S1
        S3 = (L/2 + xi)
        S4 = eta * S3
        
        return 1/L * matrix([[S1,0 ,S2,0 ,S3,0 ,S4,0],
                       [0 ,S1,0 ,S2,0 ,S3,0 ,S4]])
    
    def shapeFunctionDerivative(self,xi_,eta_):
        """

        Parameters
        ----------
        coord : INTEGER
            Number of the coordinate, x-0, y-1, z-2
        param : INTEGER
            Element parameter, xi-0, eta-1, zeta-2
        xi : DOUBLE
            Longitudinal position
        eta : lateral position

        Returns
        -------
        The derivative of the shape function evaluated at interest points.

        """
        
        L = self.length
        xi = xi_ * L/2
        eta = eta_ * self.height / 2
        
        S1 = (L/2 - xi)
        S3 = (L/2 + xi)
        
        # all the following must be scaled by 1/L. We do that in return

        dS =  np.array([[-1,0 ,-eta,0   ,1,0,eta,0]/L,
                       [0 ,-1,0   ,-eta,0,1,0  ,eta]/L,
                       [0 ,0 ,S1  ,0   ,0,0,S3 ,0]/L,
                       [0 ,0 ,0   ,S1  ,0,0,0  ,S3]/L])
        
        return dS
    
    
    def getWeightNodalForces(self,grav):
        L = self.length
        H = self.height
        W = self.width
        Qg =  L * H * W * 0.25 * dot(grav,matrix([
            [2, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 0, 0, 2, 0, 0]]))*eye(len(self.q))*self.parentBody.material.rho
        
        return Qg.reshape(1,-1)
    
    
    
    
    
#%%    
class elementQuadratic(beamANCFelement):
    """
    Planar finite element with quadratic interpolation
    """
     
    def __init__(self, node1, node2, _height, _width):
        self.length = norm(node1.qtotal[0:2] - node2.qtotal[0:2])
        self.height = _height
        self.width = _width
        intermediateNode = node()
        intermediateNode.q0 = [(a+b)*0.5 for a,b in zip(node1.q0,node2.q0)]
        #intermediateNode.qtotal = np.asarray(intermediateNode.q0) + np.asarray(intermediateNode.q)
        self.nodes = [node1,intermediateNode,node2]
        self.J0, self.invJ0, self.detJ0, self.isJ0constant = self.saveInitialJacobian()
        self.nodalElasticForces = np.zeros(12,dtype=np.float64)
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
  
    def shapeFunctionMatrix(self, xi_, eta_):
        '''
        Shape functions respect the order of the nodes: 1, intermediate, 2
        '''
        eta = eta_ * self.height / 2

        S1 = - xi_/2 * (1-xi_)
        S2 = eta * S1
        S3 = xi_/2 * (1+xi_)
        S4 = eta * S3
        S5 = 1 - xi_*xi_
        S6 = eta*S5
        
        return matrix([[S1, 0 ,S2, 0 , S5, 0 , S6, 0 , S3, 0 ,S4 ,0],
                       [0 , S1, 0, S2, 0 , S5, 0 , S6, 0 ,S3 ,0  ,S4]])
    
    def shapeFunctionDerivative(self,xi_,eta_):
        """

        Parameters
        ----------
        coord : INTEGER
            Number of the coordinate, x-0, y-1, z-2
        param : INTEGER
            Element parameter, xi-0, eta-1, zeta-2
        xi : DOUBLE
            Longitudinal position
        eta : lateral position

        Returns
        -------
        The derivative of the shape function evaluated at interest points.

        """
        
        L = self.length
        xi = xi_ * L/2
        eta = eta_ * self.height / 2
        
        
        # all the following must be scaled by 1/L^2. We do that in return
        #s1 = -L + 4*xi
        #s2 = L + 4*xi
        s1 = -1 + 2*xi_
        s2 = 1 + 2*xi_

        #dSxxi =  np.array([s1, 0, eta*s1, 0, -4*xi_, 0, -4*xi_*eta, 0, s2, 0, eta*s2, 0]) * 1/L

        #dSyxi =  np.array([0, s1, 0, eta*s1, 0, -4*xi_, 0, -4*eta*xi_, 0, s2, 0,  eta*s2]) * 1/L

        #dSxeta = np.array([0, 0, xi_/2*(-1 + xi_), 0, 0, 0, 1-xi_*xi_, 0, 0, 0, xi_/2*(1 + xi_), 0])
        
        #dSyeta = np.array([0 ,0 ,0, xi_/2*(-1 + xi_),0 ,0 ,0 ,1-xi_*xi_, 0, 0, 0, xi_/2*(1 + xi_)])

        dS = np.array([[s1, 0, eta*s1, 0, -4*xi_, 0, -4*xi_*eta, 0, s2, 0, eta*s2, 0] * 1/L,
                      [0, s1, 0, eta*s1, 0, -4*xi_, 0, -4*eta*xi_, 0, s2, 0,  eta*s2] * 1/L,
                      [0, 0, xi_/2*(-1 + xi_), 0, 0, 0, 1-xi_*xi_, 0, 0, 0, xi_/2*(1 + xi_), 0],
                      [0 ,0 ,0, xi_/2*(-1 + xi_),0 ,0 ,0 ,1-xi_*xi_, 0, 0, 0, xi_/2*(1 + xi_)]])
                      
                      
        
        return dS
    
    
    
    def getWeightNodalForces(self,grav):
        '''
        Get nodal forces due to weight
        
        TODO: currently only for initially straight beams

        Parameters
        ----------
        grav : array like
            gravity acceleration.

        Returns
        -------
        Qg : array
            nodal forces.

        '''
        L = self.length
        H = self.height
        W = self.width
        Qg =  L * H * W * 0.25 *  0.3333 * dot(grav, matrix([
            [2, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0]]))*eye(len(self.q))*self.parentBody.material.rho
        
        return Qg.reshape(1,-1)





    

            