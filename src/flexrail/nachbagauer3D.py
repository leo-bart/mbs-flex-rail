#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrapolation to 3D based on the planar beam from
K. Nachbagauer, A. Pechstein, H. Irschik, e J. Gerstmayr, “A new locking-free 
formulation for planar, shear deformable, linear and quadratic 
beam finite elements based on the absolute nodal coordinate formulation”, 
Multibody System Dynamics, vol. 26, no 3, p. 245–263, 2011,
doi: 10.1007/s11044-011-9249-8.

This version of the library is converted to Cython, in order to speed up
computation times and to generate C libraries.


Created on Mon Nov 22 07:21:00 2021

@author: Leonardo Bartalini Baruffaldi
"""

import numpy as np

from numpy.matlib import matrix, eye
from numpy import dot
from numpy.linalg import norm, inv



class node(object):
    """
    finite element node with nine dof
    
    Parameters
    __________
        listOfDof - list
            list containing the node DOF in the following order:
                listOfDof[0:3] - x,y,z coordinates of the node
                listOfDof[3:6] - x,y,z coordinates of the section slope w.r.t. eta
                listOfDof[6:9] - x,y,z coordinates of the section slope w.r.t. zeta
    """
    
    def __init__(self, listOfDof=[0.]*9):
                
        self.q0 = listOfDof
        self.q = [0.0]*len(self.q0)
        self.globalDof = [i for i in range(len(self.q0))]
        
            
        
    @property
    def q(self):
        """nodal displacement relative to reference configuration"""
        return self._q
    @q.setter
    def q(self,dofMatrix):
        self._q = dofMatrix
        
    def updateqTotal(self):
        self.qtotal = np.array(self.q, dtype=np.float64) + np.array(self.q0, dtype=np.float64)
        
    
    @property
    def qtotal(self):
        """nodal displacement relative to global frame"""
        return np.array(self.q,dtype=np.float64()) + np.array(self.q0,dtype=np.float64())
  

    
########################################
class beamANCFelement3D(object):
    '''
    Base class for three-dimensional beam finite elements
    
    TODO Make width depend on height to input rail profiles
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
    
    def interpolatePosition(self,xi_, eta_, zeta):
        """
        Returns the interpolated position given the non-dimensional parameters
        xi_, eta_, and zeta_ in [-1,1]
        """
        
        r = dot(self.shapeFunctionMatrix(xi_ ,eta_, zeta), self.qtotal)
        
        return r
    
    
    def getJacobian(self,xi_,eta_,zeta_,q=None):
        '''
        

        Parameters
        ----------
        xi_ : DOUBLE
            1st ELEMENT INSTRINSIC COORDINATE [-1,1].
        eta_ : DOUBLE
            2nd ELEMENT INSTRINSIC COORDINATE [-1,1].
        zeta_ : DOUBLE
            3rd ELEMENT INSTRINSIC COORDINATE [-1,1].
        q : VECTOR DOUBLE
            NODAL COORDINATES.

        Returns
        -------
        BLOCK MATRIX
            JACOBIAN CALCULATED AT (xi_, eta_,zeta_) under coordinates q.

        '''
        
        if q == None:
            q = self.qtotal
        nq = len(q)
  
        '''
        Shape function derivatives
        dS<i><j> = dSX_i / dxi_j
        with X_i  in [x,y,z]
             xi_j in [xi_,eta_,zeta_]
        '''
        #dSx1, dSx2, dSx3, dSy1, dSy2, dSy3, dSz1, dSz2, dSz3 = self.shapeFunctionDerivative(xi_,eta_,zeta_)
        dS = self.shapeFunctionDerivative(xi_, eta_,zeta_)
        dS = dS.reshape(3,-1)
        
        #M1 = dot([[dSx1, dSx2, dSx3],[dSy1, dSy2, dSy3],[dSz1, dSz2, dSz3]],q).round(16)
        M1 = dot([dS[:,:nq],dS[:,nq:2*nq],dS[:,2*nq:3*nq]],q).T.round(16)
        
        return np.asmatrix(M1)
    
    def saveInitialJacobian(self):
        '''
        Saves the initial jacobian on the domain edges so it does not need to
        be recomputed everytime

        Returns
        -------
        jaco : list of arrays
            list with the values of J0 at the domain boundaries.
        invJaco : list of arrays
            list with J0 inverses.
        detJaco : list of arrays
            list with J0 determinants.
        constant : boolean
            True when the Jacobian is constant.

        '''
        
        constant = np.bool8(False)
        
        jaco = []       
        
        '''create jacobian grid for computation
        first line is the front slice with xi_ = 1
        we start with eta_ = 1 and zeta_ = 1 and rotate clockwise
        
        (1,1,1) pt1 ---------- pt2 (1,1,-1)
                    |        |
                    |        |
        (1,-1,1)pt4 ---------- pt3 (1,-1,-1)
        
             ^ y
             |
         z <--
        
        '''
        jaco.append([self.initialJacobian(1,1,1), #pt1
                     self.initialJacobian(1,1,-1),#pt2
                     self.initialJacobian(1,-1,-1),#pt3
                     self.initialJacobian(1, -1, 1) #pt4
                     ])
        ''' now the backslice, same scheme with xi_=-1 '''
        jaco.append([self.initialJacobian(-1,1,1), #pt1
                     self.initialJacobian(-1,1,-1),#pt2
                     self.initialJacobian(-1,-1,-1),#pt3
                     self.initialJacobian(-1, -1, 1) #pt4
                     ])
        
        invJaco = np.linalg.inv(jaco)
        
        detJaco = np.linalg.det(jaco)
        
        #TODO adjust to used curved beams
        self.isJ0constant = True
        
        return jaco, invJaco, detJaco
        
    
    def loadInitialJacobian(self,xi_,eta_,zeta_):
        #TODO create trilinear interpolation
        
        if self.isJ0constant:
            J = self.J0[0][0]
            detJ = self.detJ0[0][0]
            invJ = self.invJ0[0][0]
            
        return J,detJ,invJ

    
    def initialJacobian(self,xi_,eta_,zeta_):
        return self.getJacobian(xi_,eta_,zeta_,self.q0)
    
    def inverseInitialJacobian(self,xi_,eta_,zeta_):
        J0 = self.initialJacobian(xi_,eta_,zeta_)
        return inv(J0)
    
    def currentJacobian(self,xi_,eta_,zeta_):
        return self.getJacobian(xi_,eta_,zeta_,self.qtotal)
    
    def getMassMatrix(self):
        
        # Gauss integration points
        gauss = self.gaussIntegrationPoints[3]
        npoints = len(gauss)
        
        # Gauss weights
        w = self.gaussWeights[3]
        
        M = 0*eye(len(self.q),dtype=np.float64)
        
        for i in range(npoints):
            for j in range(npoints):
                for k in range(npoints):
                    S = self.shapeFunctionMatrix(gauss[i],gauss[j],gauss[k])
                    M += S.T.dot(S) * w[i] * w[j] * w[k]
                
        """we have to multiply by the dimensions because
        calculations are carried out on non-dimensional coordinates [-1,1]
        """        
        return self.parentBody.material.rho * M * self.length * self.height * self.width / 8
    
    
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
                
    
    
    def stressTensorByPosition(self,xi_,eta_,zeta_,split=True):
        return self.parentBody.material.stressTensor(self.strainTensor(xi_, eta_,zeta_),split)
    
    def strainTensor(self,xi_,eta_,zeta_,q=None):
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
        
        F = self.getJacobian(xi_,eta_,zeta_,q) * self.loadInitialJacobian(xi_,eta_,zeta_)[2]
        
        return 0.5 * (dot(F.T,F) - np.eye(3))
    
    
    def shapeFunctionDerivative(self,xi_,eta_):
        raise NotImplementedError()
    
    
    
    def strainTensorDerivative(self,xi_,eta_,zeta_,q=None):
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
        invJ0 = self.loadInitialJacobian(xi_, eta_, zeta_)[2]
        
        if q == None:
            q = self.qtotal
            
        ndof = len(q)
                  
        W = self.shapeFunctionDerivative(xi_,eta_,zeta_)
        
        W = W.reshape(3,-1)
        
        Lhat = W.T.dot(W)
        Qhat = np.zeros([3,3*ndof])
        
        for i in range(3):
                Qhat[i,i*ndof:(i+1)*ndof] = q
        
        Qhat = np.dot(invJ0,Qhat)
        Qhat = np.dot(Qhat,Lhat)
        
        # TODO: separate into threads
        #U11 = np.sum([np.outer(dSx_dxi,dSx_dxi), np.outer(dSy_dxi,dSy_dxi)],axis=0)
        #U12 = np.sum([np.outer(dSx_dxi,dSx_deta), np.outer(dSy_dxi,dSy_deta)], axis=0)
        #U21 = U12.T
        #U22 = np.sum([np.outer(dSx_deta,dSx_deta), np.outer(dSy_deta,dSy_deta)],axis=0)    
        
        
        deps_dq = np.zeros((3,3,ndof),dtype=np.float64)
             

        for m in range(ndof):
            deps_dq[:,:,m] = 0.5 * Qhat[:,[m,m+ndof,m+2*ndof]].dot(invJ0)

        
       
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
        nGaussL = 2
        nGaussH = 2
        gaussL = self.gaussIntegrationPoints[nGaussL]
        gaussH = self.gaussIntegrationPoints[nGaussH]
        
        # Gauss weights
        wL = self.gaussWeights[nGaussL]
        wH = self.gaussWeights[nGaussH]
        
        
        ndof = len(self.q)
        Qe = np.zeros(ndof,dtype=np.float64)                                
        # selective reduced integration
        for p in range(nGaussL):
            'length quadrature'
            for b in range(nGaussH):
                'heigth quadrature'
                for c  in range(nGaussH):
                    detJ0 = self.loadInitialJacobian(gaussL[p], gaussH[b], gaussH[c])[1]
                    deps_dq = self.strainTensorDerivative(gaussL[p], gaussH[b],gaussH[c], q)
                    T, Tc = self.parentBody.material.stressTensor(self.strainTensor(gaussL[p], gaussH[b], gaussH[c],q),
                                                                  split=True)
                    # apply wheights to stress tensor
                    T *=detJ0 * wL[p] * wH[b] * wH[c]
                    
                    Qe += np.einsum('ij...,ij',deps_dq,T)
                    
                
            # end of height quadrature
            detJ0 = self.loadInitialJacobian(gaussL[p], 0, 0)[1]
            deps_dq = self.strainTensorDerivative(gaussL[p], 0, 0, q)
            T, Tc = self.parentBody.material.stressTensor(self.strainTensor(gaussL[p], 0, 0, q),split=True)
            
            # apply wheights to stressn tensor
            Tc *= detJ0 * wL[p]
            Qe += np.einsum('ij...,ij',deps_dq,Tc)
            #for m in range(ndof):
            #    Qe[m] += np.multiply(deps_dq[:,:,m],TcWeight).sum()
        # end of integration
            
        Qe *= W * L * H / 4
        
        self.nodalElasticForces = Qe

        return Qe
    
    
    
    def strainEnergyNorm(self):
        
        q = self.q
        
        # beam geometry
        L = self.length
        H = self.height
        W = self.width
        
        
        # Gauss integration points
        nGaussL = 3
        nGaussH = 3
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
                for c in range(nGaussH):
                    detJ0 = self.loadInitialJacobian(gaussL[p], gaussH[b], gaussH[c])[1]
                    eps = self.strainTensor(gaussL[p], gaussH[b], gaussH[c], q)
                    T, Tc = self.parentBody.material.stressTensor(eps, split=True)
                    Tweight = T * detJ0 * wL[p] * wH[b] * wH[c]
                    
                    #ts = time()
                    U += 0.5 * abs(np.einsum('ij,ij',eps,Tweight))
                    #print('{0:1.8g}'.format(time()-ts))
                    
                
            # end of height quadrature
            detJ0 = self.loadInitialJacobian(gaussL[p], 0, 0)[1]
            eps = self.strainTensor(gaussL[p], 0, 0, q)
            T, Tc = self.parentBody.material.stressTensor(eps,split=True)
            TcWeight = Tc * detJ0 * wL[p]
            for m in range(ndof):
                U += 0.5 * abs(np.multiply(eps,TcWeight).sum()) 
        # end of integration
            
        U *= W * L * H / 8
        
        return U
    
    



#%%
class beamANCF3Dlinear(beamANCFelement3D):
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

        dSxxi =  np.array([-1,0 ,-eta,0   ,1,0,eta,0])/L

        dSyxi =  np.array([0 ,-1,0   ,-eta,0,1,0  ,eta])/L

        dSxeta = np.array([0 ,0 ,S1  ,0   ,0,0,S3 ,0])/L

        dSyeta = np.array([0 ,0 ,0   ,S1  ,0,0,0  ,S3])/L
        
        return dSxxi,dSxeta, dSyxi, dSyeta
    
    
    def getWeightNodalForces(self,grav):
        L = self.length
        H = self.height
        W = self.width
        Qg =  L * H * W * 0.25 * dot(grav,matrix([
            [2, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 0, 0, 2, 0, 0]]))*eye(len(self.q))*self.parentBody.material.rho
        
        return Qg.reshape(1,-1)
    
    
    
    
    
#%%    
class beamANCF3Dquadratic(beamANCFelement3D):
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
        self.J0, self.invJ0, self.detJ0 = self.saveInitialJacobian()
        self.nodalElasticForces = np.zeros(12,dtype=np.float64)
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
  
    def shapeFunctionMatrix(self, xi_, eta_, zeta_):
        '''
        Shape functions respect the order of the nodes: 1, intermediate, 2
        '''
        eta = eta_ * self.height / 2
        zeta = zeta_ * self.height / 2
        
        #first node
        S1 = - xi_/2 * (1-xi_)
        S2 = eta * S1
        S3 = zeta * S1
        #middle node
        S4 = 1 - xi_*xi_
        S5 = eta*S4
        S6 = zeta*S4
        #last node
        S7 = xi_/2 * (1+xi_)
        S8 = eta * S7
        S9 = zeta * S7
        
        I = np.eye(3,dtype=np.float64)
        
        return np.concatenate((S1*I,S2*I,S3*I,S4*I,S5*I,S6*I,S7*I,S8*I,S9*I),
                              axis=1)
    
    def shapeFunctionDerivative(self,xi_,eta_,zeta_):
        """

        Parameters
        ----------
        xi_: DOUBLE
            Longitudinal position
        eta_: DOUBLE 
            Lateral position
        zeta_: DOUBLE
            Vertical positiion

        Returns
        -------
        List of derivatives
        
        dSx1, dSx2, dSx3, dSy1, dSy2, dSy3, dSz1, dSz2, dSz3
        
        dSx1 = dSx/dxi      dSy1 = dSy/dxi      dSz1 = dSz/dxi
        dSx2 = dSx/deta     dSy2 = dSy/deta     dSz2 = dSz/deta
        dSx3 = dSx/dzeta    dSy3 = dSy/dzeta    dSz3 = dSz/dzeta

        """
        #TODO make 3D
        L = self.length
        eta = eta_ * self.height / 2
        zeta = zeta_ * self.width / 2
        
        # reusable variables
        s1 = -1 + 2*xi_
        s2 = -4*xi_
        s3 = 1 + 2*xi_
              
        I = np.eye(3)
        
        dS = np.zeros([9,27],dtype=np.float64)
        
        # derivative wrt xi
        dS[:3,:3] = I*s1/L
        dS[:3,3:6] = I*s1*eta/L
        dS[:3,6:9] = I*s1*zeta/L
        dS[:3,9:12] = I*s2/L
        dS[:3,12:15] = I*s2*eta/L
        dS[:3,15:18] = I*s2*zeta/L
        dS[:3,18:21] = I*s3/L
        dS[:3,21:24] = I*s3*eta/L
        dS[:3,24:27] = I*s3*zeta/L
        
        # derivative wrt eta
        dS[3:6,3:6] = I*xi_*(-1+xi_)/2
        dS[3:6,12:15] = I*(1-xi_*xi_)
        dS[3:6,21:24] = I*xi_*(1+xi_)/2
        
        # derivative wrt zeta
        dS[6:9,6:9] = dS[3:6,3:6]
        dS[6:9,15:18] = dS[3:6,12:15]
        dS[6:9,24:27] = dS[3:6,21:24]
        
        return dS
    
    
    
    def getWeightNodalForces(self,grav=[0,-1.,0]):
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
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]],
            dtype=np.float64))*eye(len(self.q))*self.parentBody.material.rho
        
        return Qg.reshape(1,-1)





if __name__ == '__main__':
    print('Unit test')
    import materials, flexibleBody
    steel = materials.linearElasticMaterial('Steel',200e9,0.3,7.85e3)
    rod = flexibleBody.flexibleBody('Rod',steel)

    # comprimentos
    Lrod = 100.0e-3

    # seção transversal
    h = 10.0e-3

    # gravidade
    g = [0,-9.810]

    # malha
    nel = 2
    nodesA = [0]*(nel+1)
    for i in range(nel+1):
        nodesA[i]=node([Lrod * i/nel,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
        
    elemA = [0]*nel
    for i in range(nel):
        elemA[i]=beamANCF3Dquadratic(nodesA[i],nodesA[i+1], h, h)
        
    rod.addElement(elemA)
    M = rod.assembleMassMatrix()
    G = rod.assembleWeightVector(g=[0,1,0])
    
    elemA[0].strainTensorDerivative(1,0,0)
    
    # force deformation
    elemA[1].nodes[2].q[0] = 0.01e-3
    elemA[1].nodes[1].q[0] = 0.01e-3 * 0.75
    elemA[1].nodes[0].q[0] = 0.01e-3 * 0.50
    elemA[0].nodes[1].q[0] = 0.01e-3 * 0.25

    F = rod.assembleElasticForceVector()


    

            