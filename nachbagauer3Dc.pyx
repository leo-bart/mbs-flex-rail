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
from MultibodySystem import marker
from numpy.matlib import matrix, eye
from numpy import dot
from numpy.linalg import norm, inv
cimport cython

@cython.boundscheck(False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         3D NODE                                                             %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
cdef public class node[object c_PyObj, type c_PyObj_t]:    
    
    cdef double [9] q0
    cdef double [9] q
    cdef double [9] u0
    cdef double [9] u
    cdef long [9] globalDof
    cdef public str name
    cdef object marker
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
                
        self.name = 'Node'
        self.q0 = np.array(listOfDof, dtype=np.float64)
        self.q = 0 * np.array(listOfDof, dtype=np.float64)
        self.globalDof = np.arange(9, dtype=np.int64)
        self.marker = marker('_node', self.qtotal[:3], self.orientation)
        self.marker.setParent(self)
        
            
    '''Class properties'''
    
    @property
    def q(self):
        '''
        nodal dof
        '''
        return np.array(self.q,dtype=np.float64)
    @q.setter
    def q(self,dofMatrix):
        self.q = dofMatrix
  
    @property
    def q0(self):
        '''
        nodal initial positions
        '''
        return np.array(self.q0, dtype=np.float64)
    @q0.setter
    def q0(self,dofMatrix):
        self.q0 = dofMatrix
        
    @property
    def u(self):
        '''
        nodal velocities
        '''
        return np.array(self.u, dtype=np.float64)
    @u.setter
    def u(self,uMatrix):
        self.u = uMatrix
        
    @property
    def u0(self):
        '''
        nodal initial velocities
        '''
        return np.array(self.u0, dtype=np.float64)
    @u0.setter
    def u0(self,uMatrix):
        self.u0 = uMatrix
        
    @property
    def globalDof(self):
        '''
        globalDof number
        '''
        return np.array(self.globalDof, dtype=np.int64)
    @globalDof.setter
    def globalDof(self,dofList):
        self.globalDof = dofList
              
    @property
    def qtotal(self):
        '''Total nodal position'''
        return np.array(self.q) + np.array(self.q0)
    
    @property
    def orientation(self):
        cdef double [:] e2, e3
        cdef Py_ssize_t i
        
        e2 = self.qtotal[3:6]
        e3 = self.qtotal[6:9]
        
        e1 = np.cross(e2,e3)
            
        return np.vstack([e1,e2,e3]).transpose()
    
    @property 
    def marker(self):
        return self.marker


    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         GENERIC 3D ELEMENT                                                  %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
cdef class beamANCFelement3D(object):
    '''
    Base class for three-dimensional beam finite elements
    
    TODO Make width depend on height to input rail profiles
    '''
    
    ''' Properties' declarations'''
    
    cdef object parentBody
    cdef double length
    cdef double height
    cdef double width
    cdef double[:] nodalElasticForces
    cdef double[:,:] detJ0
    cdef double[:,:,:,:] J0, invJ0
    cdef bint changedStates, isJ0constant
    cdef list nodes
    
    
    def __init__(self):
        self.parentBody = None
        self.length = 1.0
        self.height = 1.0
        self.width = 1.0
        self.nodalElasticForces = None
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
        
        
       
    
    @property
    def q0(self):
        q0n = []
        for n in self.nodes:
            q0n.extend(n.q0)
        return np.array(q0n)
    
    @property
    def gaussIntegrationPoints(self):
        return {1:[0],
                2:[-1.0/np.sqrt(3),1.0/np.sqrt(3)],
                3:[-0.77459667, 0, 0.77459667],
                4:[-0.86113631,-0.33998104,0.33998104,0.86113631],
                5:[-0.90617985,-0.53846931,0.0,0.53846931,0.90617985],
                6:[-0.93246951,-0.66120938,-0.23861919,0.23861919,0.66120938,0.93246951]}
    
    @property
    def lobattoIntegrationPoints(self):
        return {4:[-1.,-0.333333,0.333333,1.]}
    
    @property
    def gaussWeights(self): 
        return {1:[2],
                2:[1,1],
                3:[0.5555556, 0.8888889, 0.5555556],
                4:[0.3478548,0.6521452,0.6521452,0.3478548],
                5:[0.2369269,0.4786287,0.5688889,0.4786287,0.2369269],
                6:[0.1713245,0.3607616,0.4679139,0.4679139,0.3607616,0.1713245]}
    
    @property
    def lobattoWeights(self):
        return {4:[0.125,0.375,0.375,0.125]}
    
    @property
    def qtotal(self):
        """nodal position relative to global frame"""
        myq = []
        for node in self.nodes:
            myq.extend(node.qtotal.tolist())  
        return np.array(myq, dtype=np.float64)
    
    @property
    def q(self):
        '''nodal displacement relative to global frame'''
        myq = []
        for node in self.nodes:
            myq.extend(node.q)
            
        return np.array(myq, dtype=np.float64)
    
    @property
    def u(self):
        '''nodal velocities relative to global frame'''
        myu = []
        for node in self.nodes:
            myu.extend(node.u)
            
        return np.array(myu, dtype=np.float64)
    
    @property
    def globalDof(self):
        gd = []
        for nd in self.nodes:
            gd.extend(nd.globalDof)
        return np.array(gd, dtype=np.int64)
    
    @property
    def nodes(self):
        return self.nodes
    @nodes.setter
    def nodes(self,listOfNodes):
        self.nodes = listOfNodes
        
    @property
    def parentBody(self):
        return self.parentBody
    @parentBody.setter
    def parentBody(self,_body):
        self.parentBody = _body
        
    @property
    def changedStates(self):
        return self.changedStates
    @changedStates.setter
    def changedStates(self, bint state):
        self.changedStates = state
    
    @property
    def mass(self):
        return self.length * self.height * self.width * self.parentBody.material.rho
    
    
    def getBoundingBox(self):
        '''
        Get element bounding box. Usefull to check for contact points.
        
        TODO: need to consider width and height too
        
        Based on the idea from  
        G. H. C. Silva, R. Le Riche, J. Molimard, e A. Vautrin, 
        “Exact and efficient interpolation using finite
        elements shape functions”, 
        European Journal of Computational Mechanics, vol. 18, nº 3–4, 
        p. 307–331, jan. 2009, doi: 10.3166/ejcm.18.307-331.


        Returns
        -------
        coords : list
            list min and max values for nodes coordinates

        '''
        cdef double xmin, xmax, ymin, ymax, zmin, zmax, x, y, z
        cdef Py_ssize_t i,j,k
        cdef list virtualNodes = []
        
        for i in [-1,1]:
            for j in [-1,1]:
                for k in [-1,1]:
                    virtualNodes.append(self.interpolatePosition(i,j,k))
        
        xmin, ymin, zmin = virtualNodes[0].tolist()
        
        xmax = xmin
        ymax = ymin
        zmax = zmin
        
        for i in range(1,len(virtualNodes)):
            x, y, z = virtualNodes[i].tolist()
            if xmin > x:
                xmin = x
            if xmax < x:
                xmax = x
            if ymin > y:
                ymin = y
            if ymax < y:
                ymax = y
            if zmin > z:
                zmin = z
            if zmax < z:
                zmax = z
        
        return xmin, ymin, zmin, xmax, ymax, zmax
        
        
    
    
    def isPointOnMe(self, double[:] P):
        '''
        Check whether a point P belongs to the object bounding box

        Parameters
        ----------
        double[ : ] P
            DESCRIPTION.

        Returns
        -------
        ispom : boolean
            returns True if the point is inside de bounding box.

        '''
        cdef double xmin, ymin, zmin, xmax, ymax, zmax
        cdef tol = 1e-4 * self.length
        
        xmin, ymin, zmin, xmax, ymax, zmax = self.getBoundingBox()
        
        if P[0] > xmax + tol:
            return False
        elif P[1] > ymax + tol:
            return False
        elif P[2] > zmax + tol:
            return False
        elif P[0] < xmin - tol:
            return False
        elif P[1] < ymin - tol:
            return False
        elif P[2] < zmin - tol:
            return False
        else:
            # TODO: check cross product before issuing True
            return True
    
    def mapToLocalCoords(self, double[:] point, double tol = 1e-5):
        '''
        Maps a global point P into local coordinates

        Parameters
        ----------
        double[ : ] point
            global coordinates of a point that is to be mapped locally.

        Returns
        -------
        localP : array
            local coordinates of the point.

        '''
        
        if not self.isPointOnMe(point):
            print('Error: specified point is not inside the bounding box of this element')
            return 0
        
        cdef double L,H,W
        cdef Py_ssize_t i
        cdef long maxiter
        
        # initialize local variables
        maxiter = 20
        
        L = self.length
        H = self.height
        W = self.width
        
        xi_view = np.zeros(3)
        dxi = xi_view
        p = np.array(point)
        
        for i in range(maxiter):
            xi_view += dxi
                       
            rn = self.interpolatePosition(xi_view[0],xi_view[1],xi_view[2])
            res = p - rn
            if np.all(np.abs(res)<tol):
                break

            J_view = self.getJacobian(xi_view[0],xi_view[1],xi_view[2]).reshape(3,-1)
            # scaling factors:
            J_view[0] *= L/2
            J_view[1] *= H/2
            J_view[2] *= W/2
            
            dxi = np.linalg.solve(J_view,res)
        
        
        return xi_view
        
    
    
    def interpolatePosition(self,double xi_, double eta_, double zeta_):
        """
        Returns the interpolated position given the non-dimensional parameters
        xi_, eta_, and zeta_ in [-1,1]
        """
        
        r = dot(self.shapeFunctionMatrix(xi_ ,eta_, zeta_), self.qtotal)
        
        return r
    
    
    def interpolateVelocity(self,double xi_, double eta_, double zeta_):
        """
        Returns the interpolated velocity given the non-dimensional parameters
        xi_, eta_, and zeta_ in [-1,1]
        """
        return self.shapeFunctionMatrix(xi_ ,eta_, zeta_).dot(self.u)
    
    
    
    def getJacobian(self, double xi_, double eta_, double zeta_, double [:] q = None):
        '''
        Jacobian of the absolute position vector dr/dxi
        

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
        Matrix
            JACOBIAN CALCULATED AT (xi_, eta_,zeta_) under nodal coordinates q.

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
        
        dS = self.shapeFunctionDerivative(xi_, eta_,zeta_).reshape(3,-1)
        
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
        
        jaco = np.array(jaco)
        
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

    
    def initialJacobian(self,double xi_, double eta_, double zeta_):
        return self.getJacobian(xi_,eta_,zeta_,self.q0)
    
    def inverseInitialJacobian(self,double xi_, double eta_, double zeta_):
        J0 = self.initialJacobian(xi_,eta_,zeta_)
        return inv(J0)
    
    def currentJacobian(self,double xi_, double eta_, double zeta_):
        return self.getJacobian(xi_,eta_,zeta_,self.qtotal)
    
    def getMassMatrix(self):
        
        # Gauss integration points
        cdef list gauss = self.gaussIntegrationPoints[3]
        cdef Py_ssize_t npoints = len(gauss)
        
        # Gauss weights
        cdef list w = self.gaussWeights[3]
        
        cdef Py_ssize_t msize = len(self.q)
        
        M = np.zeros((msize,msize),dtype=np.float64)
    
        cdef Py_ssize_t i,j, k
        
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
        
        cdef double[:] qmod
        cdef Py_ssize_t i
        cdef double curDof
        
        for nd in self.nodes:
            qmod = nd.q.copy()
            for i,curDof in enumerate(nd.q):
                savePos = curDof
                qmod[i] += 1e-6
                nd.q = qmod
                Kte[:,col] = (self.getNodalElasticForces() - Q0) * 1e6
                qmod[i] = savePos
                nd.q = qmod
                col += 1
                
        return Kte
                
    
    
    def stressTensorByPosition(self,double xi_,double eta_,double zeta_,bint split=True):
        return self.parentBody.material.stressTensor(self.strainTensor(xi_, eta_,zeta_),split)
    
    def strainTensor(self,double xi_,double eta_,double zeta_,double [:] q=None):
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
    
    
    
    cdef strainTensorDerivative(self,double xi_,double eta_,double zeta_,double [:] q=None):
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
        invJ0 = np.array(self.loadInitialJacobian(xi_, eta_, zeta_)[2])
        
        if q == None:
            q = self.qtotal
            
        ndof = len(q)
                  
        W = self.shapeFunctionDerivative(xi_,eta_,zeta_)
        W = W.reshape(3,-1)
        
        Lhat = W.T.dot(W)
        Qhat = np.zeros([3,3*ndof])
        cdef double[:,:] Qhat_view = Qhat
        
        cdef Py_ssize_t i
        
        for i in range(3):
                Qhat_view[i,i*ndof:(i+1)*ndof] = q

        Qhat = invJ0.dot(Qhat)
        Qhat = Qhat.dot(Lhat)
        
        # TODO: separate into threads
        #U11 = np.sum([np.outer(dSx_dxi,dSx_dxi), np.outer(dSy_dxi,dSy_dxi)],axis=0)
        #U12 = np.sum([np.outer(dSx_dxi,dSx_deta), np.outer(dSy_dxi,dSy_deta)], axis=0)
        #U21 = U12.T
        #U22 = np.sum([np.outer(dSx_deta,dSx_deta), np.outer(dSy_deta,dSy_deta)],axis=0)    
        
        
        deps_dq = np.zeros((3,3,ndof),dtype=np.float64)
             
        cdef Py_ssize_t m
        
        for m in range(ndof):
            deps_dq[...,m] = 0.5 * Qhat[:,[m,m+ndof,m+2*ndof]].dot(invJ0)

        
       
        return deps_dq 
    
    
   
    def getNodalElasticForces(self,bint veloc = False):
        
        
        # beam geometry
        cdef double L = self.length
        cdef double H = self.height
        cdef double W = self.width
        
        # TODO correct changedStates calculation
   
        if veloc:
            q = self.u
        else:
            q = self.qtotal
        
        # Gauss integration points
        cdef long nGaussL = 2
        cdef long nGaussH = 2
        cdef list gaussL = self.gaussIntegrationPoints[nGaussL]
        cdef list gaussH = self.gaussIntegrationPoints[nGaussH]
        
        # Gauss weights
        cdef list wL = self.gaussWeights[nGaussL]
        cdef list wH = self.gaussWeights[nGaussH]
        
        
        cdef long ndof = len(self.q)
        Qe = np.zeros(ndof,dtype=np.float64)

        cdef double[:,:] T, Tc
        cdef double[:,:,:] deps_dq
        cdef double detJ0
        
        cdef Py_ssize_t p,b,c
                              
        # selective reduced integration
        for p in range(nGaussL):
            'length quadrature'
            for b in range(nGaussH):
                'heigth quadrature'
                for c  in range(nGaussH):
                    detJ0 = self.loadInitialJacobian(gaussL[p], gaussH[b], gaussH[c])[1]
                    deps_dq = self.strainTensorDerivative(gaussL[p], gaussH[b],gaussH[c], q)
                    # integration weights get applied to stress tensor in the following
                    T = self.parentBody.material.stressTensor(
                        self.strainTensor(gaussL[p], gaussH[b], gaussH[c],q),
                        split=True)[0] * detJ0 * wL[p] * wH[b] * wH[c]
                    
                    Qe += np.einsum('ij...,ij',deps_dq,T)
                    
                
            # end of height quadrature
            detJ0 = self.loadInitialJacobian(gaussL[p], 0, 0)[1]
            deps_dq = self.strainTensorDerivative(gaussL[p], 0, 0, q)
            # integration weights get applied to stress tensor in the following
            Tc = self.parentBody.material.stressTensor(
                self.strainTensor(gaussL[p], 0, 0, q),split=True)[1] * detJ0 * wL[p]
            
            Qe += np.einsum('ij...,ij',deps_dq,Tc)
            #for m in range(ndof):
            #    Qe_view[m] += np.multiply(deps_dq[:,:,m],Tc).sum()
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
    
    



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         LINEAR ELEMENT                                                      %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
cdef class beamANCF3Dlinear(beamANCFelement3D):
    
    
    
    '''
    Planar finite element with linear interpolation
    
    TODO:  finish conversion to 3D
    
    '''
     
    def __init__(self, node1, node2, _height, _width):
        self.length = norm(node1.qtotal[0:2] - node2.qtotal[0:2])
        self.height = _height
        self.width = _width
        self.nodes = [node1, node2]
        self.J0, self.invJ0, self.detJ0, self.isJ0constant = self.saveInitialJacobian()
        self.nodalElasticForces = np.zeros(18,dtype=np.float64)
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
        
    
    

    def shapeFunctionMatrix(self, double xi_, double eta_, double zeta_):

        cdef double L = self.length
        cdef double xi = xi_ * L/2
        cdef double eta = eta_ * self.height / 2
        cdef double zeta = zeta_ * self.width/ 2

        cdef double S1 = (L/2 - xi)
        cdef double S2 = eta * S1
        cdef double S3 = (L/2 + xi)
        cdef double S4 = eta * S3
        
        return 1/L * np.array([[S1,0 ,S2,0 ,S3,0 ,S4,0],
                       [0 ,S1,0 ,S2,0 ,S3,0 ,S4]])
    
    def shapeFunctionDerivative(self, double xi_, double eta_, double zeta_):
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
        
        cdef double L = self.length
        cdef double xi = xi_ * L/2
        cdef double eta = eta_ * self.height / 2
        
        cdef double S1 = (L/2 - xi)
        cdef double S3 = (L/2 + xi)
        
        # all the following must be scaled by 1/L. We do that in return

        dSxxi =  [-1,0 ,-eta,0   ,1,0,eta,0]

        dSyxi =  [0 ,-1,0   ,-eta,0,1,0  ,eta]

        dSxeta = [0 ,0 ,S1  ,0   ,0,0,S3 ,0]

        dSyeta = [0 ,0 ,0   ,S1  ,0,0,0  ,S3]
        
        return np.array([[dSxxi,dSxeta], [dSyxi, dSyeta]]) / L
    
    
    def getWeightNodalForces(self, double[:] grav):
        cdef double L = self.length
        cdef double H = self.height
        cdef double W = self.width
        Qg =  L * H * W * 0.25 * dot(grav,matrix([
            [2, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 0, 0, 2, 0, 0]]))*eye(len(self.q))*self.parentBody.material.rho
        
        return Qg.reshape(1,-1)
    
    
    
    
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         QUADRATIC ELEMENT                                                   %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
cdef class beamANCF3Dquadratic(beamANCFelement3D):
    """
    Planar finite element with quadratic interpolation
    """
     
    def __init__(self, node1, node2, double _height, double _width):
        self.length = norm(node1.qtotal[0:2] - node2.qtotal[0:2])
        self.height = _height
        self.width = _width
        intermediateNode = node()
        intermediateNode.q0 = np.array([(a+b)*0.5 for a,b in zip(node1.q0,node2.q0)])
        self.nodes = [node1,intermediateNode,node2]
        self.J0, self.invJ0, self.detJ0 = self.saveInitialJacobian()
        self.nodalElasticForces = np.zeros(27,dtype=np.float64)
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
  
    def shapeFunctionMatrix(self, double xi_, double eta_, double zeta_):
        '''
        Shape functions respect the order of the nodes: 1, intermediate, 2
        '''
        cdef double eta = eta_ * self.height / 2
        cdef double zeta = zeta_ * self.width / 2
        
        #first node
        cdef double S1 = - xi_/2 * (1-xi_)
        cdef double S2 = eta * S1
        cdef double S3 = zeta * S1
        #middle node
        cdef double S4 = 1 - xi_*xi_
        cdef double S5 = eta*S4
        cdef double S6 = zeta*S4
        #last node
        cdef double S7 = xi_/2 * (1+xi_)
        cdef double S8 = eta * S7
        cdef double S9 = zeta * S7
        
        I = np.eye(3,dtype=np.float64)
        
        return np.concatenate((S1*I,S2*I,S3*I,S4*I,S5*I,S6*I,S7*I,S8*I,S9*I),
                              axis=1)
    
    def shapeFunctionDerivative(self,double xi_, double eta_, double zeta_):
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
        cdef double L = self.length
        cdef double eta = eta_ * self.height / 2
        cdef double zeta = zeta_ * self.width / 2
        
        # reusable variables
        cdef double s1,s2,s3 
        s1 = (-1 + 2*xi_)/L
        s2 = (-4*xi_)/L
        s3 = (1 + 2*xi_)/L
              
        I = np.eye(3)
        
        dS = np.zeros([9,27],dtype=np.float64)
        cdef double[:,:] dS_view = dS
        
        
        cdef Py_ssize_t i
        for i in range(3):
                dS_view[i,i] = s1
                dS_view[i,i+3] = s1*eta
                dS_view[i,i+6] = s1*zeta
                dS_view[i,i+9] = s2
                dS_view[i,i+12] = s2*eta
                dS_view[i,i+15] = s2*zeta
                dS_view[i,i+18] = s3
                dS_view[i,i+21] = s3*eta
                dS_view[i,i+24] = s3*zeta
                
                # derivative wrt eta
                dS_view[i+3,i+3] = - xi_*(-1+xi_)/2
                dS_view[i+3,i+12] = -(1-xi_*xi_)
                dS_view[i+3,i+21] = - xi_*(1+xi_)/2
                
                # derivative wrt zeta
                dS_view[i+6,i+6] = dS_view[i+3,i+3]
                dS_view[i+6,i+15] = dS_view[i+3,i+12]
                dS_view[i+6,i+24] = dS_view[i+3,i+21]
        

        
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
        cdef double L = self.length
        cdef double H = self.height
        cdef double W = self.width
        Qg =  L * H * W * 0.25 *  0.3333 * dot(grav, matrix([
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]],
            dtype=np.float64))*eye(len(self.q))*self.parentBody.material.rho
        
        return Qg.reshape(1,-1)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#         QUADRATIC RAIL ELEMENT                                              %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
cdef class railANCF3Dquadratic(beamANCF3Dquadratic):
    """
    Planar finite element with quadratic interpolation adapted to TR-68 rail
    """
    
    
    ''' Properties' declarations'''
    
    cdef double centroidHeightFromBase
    cdef double baseHeight, headHeight, webHeight
    cdef double baseWidth, headWidth, webWidth
    cdef double crossSecArea
     
    def __init__(self, node1, node2, 
                 double _height, 
                 double _width, 
                 double _hc,
                 double _hb,
                 double _hh,
                 double _wb,
                 double _ww,
                 double _wh,
                 double _cSecArea):
        '''
        

        Parameters
        ----------
        node1 : node
            First element node.
        node2 : node
            Second element node.
        double _height : double
            Total height of the cross section.
        double _width : double
            Maximum width of the cross section.
        double _hc : double
            Centroid height
        double _hb : double
            Rail base height.
        double _hh : double
            Rail head height.
        double _wb : double
            Rail base width
        double _ww : double
            Rail web width
        double _wh : double
            Rail head width
        double _cSecArea : double
            Rail section area (from standards)

        Returns
        -------
        None.

        '''
        self.length = norm(node1.qtotal[0:2] - node2.qtotal[0:2])
        self.height = _height
        # rail width varies with height
        self.width = _width
        self.baseHeight = _hb
        self.headHeight = _hh
        self.webHeight = _height - _hb - _hh
        self.baseWidth = _wb
        self.webWidth = _ww
        self.headWidth = _wh
        # centroid height
        self.centroidHeightFromBase = _hc
        self.crossSecArea = _cSecArea
        intermediateNode = node()
        intermediateNode.q0 = np.array([(a+b)*0.5 for a,b in zip(node1.q0,node2.q0)])
        self.nodes = [node1,intermediateNode,node2]
        self.J0, self.invJ0, self.detJ0 = self.saveInitialJacobian()
        self.nodalElasticForces = np.zeros(27,dtype=np.float64)
        # boolean flag that checks if the state had changed recently
        self.changedStates = np.bool8(True) 
        
 
    def getWidth(self, double eta_ = -1):
        # based on an equivalent section to TR68 rail

        cdef double heiFromFoot = self.height / 2 * (1 + eta_)
         
        if heiFromFoot < self.baseHeight:
            return self.baseWidth
        elif heiFromFoot < (self.baseHeight + self.webHeight):
            return self.webWidth
        else:
            return self.headWidth
        
        
    def getNodalElasticForces(self,bint veloc = False):
        '''
        Overrides base class method

        Parameters
        ----------
        double [ : ] q, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Qe : TYPE
            DESCRIPTION.

        '''
        
        # beam geometry
        cdef double L = self.length
        cdef double H = self.height
        cdef double W = self.width
        cdef double WB = 135.605e-3
        cdef double WW = 23.815e-3
        cdef double WH = 78.339e-3
        cdef double HB = self.baseHeight
        cdef double HW = self.webHeight
        cdef double HH = self.headHeight
        cdef double area = self.crossSecArea
        cdef double etaHc = 2 * self.centroidHeightFromBase / H - 1       # centroid height in element specficic coordinates
        cdef double etaHb = 2 * self.baseHeight / H - 1
        cdef double etaHw = 2 * self.baseHeight / H - 4 * self.webHeight - 1
        cdef double etaHh = - 2 * self.headHeight / H + 1
        
        # TODO use composite quadrature rules to integrate over different sections
   
        if veloc:
            q = self.u
        else:
            q = self.qtotal
        
        # Gauss integration points
        cdef long nGaussL = 2
        cdef long nGaussH = 2
        cdef long nGaussW = 2
        
        cdef list gaussL = self.gaussIntegrationPoints[nGaussL]
        cdef list gaussH = self.gaussIntegrationPoints[nGaussH]
        cdef list gaussW = self.gaussIntegrationPoints[nGaussW]
        
        # height of the sections
        cdef double etaBase = self.baseHeight / self.height
        cdef double etaWeb = self.webHeight / self.height
        cdef double etaHead = self.headHeight / self.height
              
        # Gauss weights
        cdef list wL = self.gaussWeights[nGaussL]
        cdef list wH = self.gaussWeights[nGaussH]
        cdef list wW = self.gaussWeights[nGaussW]
        
        
        cdef long ndof = len(self.q)
        Qe = np.zeros(ndof,dtype=np.float64)

        # cdef double[:,:] T, Tc
        # cdef double[:,:,:] deps_dq
        # cdef double detJ0
        intPoints = np.empty(3)
        weights = np.empty(3)
        cdef double[:] ip_v, w_v
        ip_v = intPoints
        w_v = weights
        
        cdef Py_ssize_t p,b,c
                              
        # selective reduced integration
        for p in range(nGaussL):
            'length quadrature'
            for c in range(nGaussW):
                'width quadrature'
                for b  in range(nGaussH):  
                    'height quadrature'
                    ip_v[0] = gaussL[p]
                    ip_v[2] = gaussW[c]
                    
                    w_v[0] = wL[p]
                    w_v[1] = wH[b]
                    w_v[2] = wW[c]
                    
                    # TODO create one ip_v for each section
                    # BASE SECTION
                    ip_v[1] = gaussH[b] * 2*HB/H + etaHb
                    W = WB
                    Qe += self.forceAtIntegrationPoint(
                        ip_v,
                        w_v,q,0) * W * (H-HB) / 4
                    
                    # BASE SECTION
                    ip_v[1] = gaussH[b] * 2*HW/H + etaHw
                    W = WW
                    Qe += self.forceAtIntegrationPoint(
                        ip_v,
                        w_v,q,0) * W * HW / 4
                    
                    # BASE SECTION
                    ip_v[1] = gaussH[b] * 2 * HH/H + etaHh
                    W = WH
                    Qe += self.forceAtIntegrationPoint(
                        ip_v,
                        w_v,q,0) * W * HH / 4
                    # end of height integration
                # end of width integration
            # end of height integration
            ip_v[0] = gaussL[p]
            ip_v[1] = etaHc
            ip_v[2] = 0.0
            
            w_v[0] = wL[p]
            w_v[1] = 1.0
            w_v[2] = 1.0
            
            Qe += self.forceAtIntegrationPoint(
                ip_v,
                w_v,q,1) * area

        # end of integration
            
        Qe *= L / 2
        
        self.nodalElasticForces = Qe

        return Qe
    
    def forceAtIntegrationPoint(self, double[:] intPoint, 
                                double[:] weights, double [:] q, 
                                int shear=0):
        '''
        Computes nodal internal forces for a specific point in
        element coordinates. The force must be scaled using the 
        appropriate lenghts.

        Parameters
        ----------
        double[ : ] intPoint
            Integration point with up to three coordinates.
        double[ : ] weights
            Integration weights corresponding to each integration point coordinate.
        double [ : ] q
            Current nodal positions.
        int shear : TYPE, optional
            Selects between bending and shear integration. The default is 0.

        Returns
        -------
        np.array Qe
            Vector of (partial) nodal forces.

        '''
        cdef double[:,:] T, Tc
        cdef double[:,:,:] deps_dq
        cdef double detJ0
        
        detJ0 = self.loadInitialJacobian(intPoint[0],intPoint[1],intPoint[2])[1]
        deps_dq = self.strainTensorDerivative(intPoint[0],intPoint[1],intPoint[2],q)
        T = self.parentBody.material.stressTensor(
            self.strainTensor(intPoint[0],intPoint[1],intPoint[2],q),
            split=True)[shear] * detJ0 * weights[0] * weights[1] * weights[2]
        
        return np.einsum('ij...,ij',deps_dq,T)

  
    def shapeFunctionMatrix(self, double xi_, double eta_, double zeta_):
        '''
        Shape functions respect the order of the nodes: 1, intermediate, 2
        '''
        cdef double eta = eta_ * self.height / 2
        cdef double zeta = zeta_ * self.getWidth(eta_) / 2
        
        #first node
        cdef double S1 = - xi_/2 * (1-xi_)
        cdef double S2 = eta * S1
        cdef double S3 = zeta * S1
        #middle node
        cdef double S4 = 1 - xi_*xi_
        cdef double S5 = eta*S4
        cdef double S6 = zeta*S4
        #last node
        cdef double S7 = xi_/2 * (1+xi_)
        cdef double S8 = eta * S7
        cdef double S9 = zeta * S7
        
        I = np.eye(3,dtype=np.float64)
        
        return np.concatenate((S1*I,S2*I,S3*I,S4*I,S5*I,S6*I,S7*I,S8*I,S9*I),
                              axis=1)
    
    def shapeFunctionDerivative(self,double xi_, double eta_, double zeta_):
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
        cdef double L = self.length
        cdef double eta = eta_ * self.height / 2
        cdef double zeta = zeta_ * self.getWidth(eta_) / 2
        
        # reusable variables
        cdef double s1,s2,s3 
        s1 = (-1 + 2*xi_)/L
        s2 = (-4*xi_)/L
        s3 = (1 + 2*xi_)/L
              
        I = np.eye(3)
        
        dS = np.zeros([9,27],dtype=np.float64)
        cdef double[:,:] dS_view = dS
        
        
        cdef Py_ssize_t i
        for i in range(3):
                dS_view[i,i] = s1
                dS_view[i,i+3] = s1*eta
                dS_view[i,i+6] = s1*zeta
                dS_view[i,i+9] = s2
                dS_view[i,i+12] = s2*eta
                dS_view[i,i+15] = s2*zeta
                dS_view[i,i+18] = s3
                dS_view[i,i+21] = s3*eta
                dS_view[i,i+24] = s3*zeta
                
                # derivative wrt eta
                dS_view[i+3,i+3] = xi_*(-1+xi_)/2
                dS_view[i+3,i+12] = (1-xi_*xi_)
                dS_view[i+3,i+21] = xi_*(1+xi_)/2
                
                # derivative wrt xeta
                dS_view[i+6,i+6] = dS_view[i+3,i+3]
                dS_view[i+6,i+15] = dS_view[i+3,i+12]
                dS_view[i+6,i+24] = dS_view[i+3,i+21]
        

        
        return dS
    
    def getMassMatrix(self):
        
        # Gauss integration points
        cdef list gauss = self.gaussIntegrationPoints[3]
        cdef Py_ssize_t npoints = len(gauss)
        
        # Gauss weights
        cdef list w = self.gaussWeights[3]
        
        cdef Py_ssize_t msize = len(self.q)
        
        M = np.zeros((msize,msize),dtype=np.float64)
    
        cdef Py_ssize_t i,j, k
        
        for i in range(npoints):
            for j in range(npoints):
                for k in range(npoints):
                    S = self.shapeFunctionMatrix(gauss[i],gauss[j],gauss[k])
                    M += S.T.dot(S) * w[i] * w[j] * w[k]
                
        """we have to multiply by the dimensions because
        calculations are carried out on non-dimensional coordinates [-1,1]
        """        
        return self.parentBody.material.rho * M * self.length * self.height * self.width / 8
    
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
        cdef double L = self.length
        cdef double H = self.height
        cdef double W = self.width
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