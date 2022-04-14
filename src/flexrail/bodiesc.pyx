#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:55:42 2022

Classes to use with Multibody System as bodies

@author: leonardo
"""

import numpy as np
import matplotlib.pyplot as plt
import MultibodySystem as mbs
import helper_funcs as hf

cdef class body(object):
    
    cdef str name, type
    cdef int totalDof
    cdef double mass
    cdef double[:,:] massMatrix
    cdef double[:] q0, u0
    cdef double[:,:] simQ, simU, simF
    cdef public list globalDof
    cdef list markers, profiles
    
    def __init__(self,name_,numberOfDof=0):
        self.name = name_
        self.type = 'Generic body descriptor'
        self.totalDof = numberOfDof
        self.q0 = np.zeros(self.totalDof, dtype=np.float64)
        self.u0 = np.zeros(self.totalDof, dtype=np.float64)
        self.globalDof = []
        self.markers = []
        self.profiles = []
    
    @property
    def name(self):
        return self.name
    
    @property
    def type(self):
        return self.type
    
    @property
    def markers(self):
        return self.markers
    
    @property
    def profiles(self):
        return self.profiles
    
    @property 
    def mass(self):
        return self.mass
    
    @property
    def totalDof(self):
        return self.totalDof
    
    @property
    def q0(self):
        return np.array(self.q0)
    
    @property
    def u0(self):
        return np.array(self.u0)    
    
    @property
    def simQ(self):
        return np.array(self.simQ)
    @simQ.setter
    def simQ(self, simArray):
        self.simQ = simArray
    
    @property
    def simU(self):
        return np.array(self.simU)
    @simU.setter
    def simU(self, simArray):
        self.simU = simArray
    
    
    
    
    
    ####### METHODS ##################################
    def addMarker(self, mrk):
        '''
        Adds a markr to the body markers list.
        
        The added marker parentBody property is automalically set to the body
        it is added to.

        Parameters
        ----------
        mrk : marker
            The marker to be added.

        Returns
        -------
        mrk : marker
            The marker created .

        '''
        if type(mrk) is list:
            print('{}.addMarker error: expected single marker as input, not list')
        
        self.markers.append(mrk)
        mrk.setParent(self)
        
        return mrk
    
    def addProfile(self, prof):
        '''
        Adds a profile to the body

        Parameters
        ----------
        prof : profile
            The profile to be added.

        Returns
        -------
        prof : profile
            The profile that has been added. It is returned to be used in other
            parts of the code, if needed.

        '''
        self.profiles.append(prof)
        prof.setParent(self)
        
        return prof
    
    def setPositionInitialConditions(self,*args):
        '''
        Set the initial conditions on position level
        
        This function has two possible calls
        
        setPositionInitialConditions(q) expects q to be an array with all dofs specified
        
        setPositionInitialConditions(qInd,val) expects qInd to be the index of the dof and val is the initial value attributed to q[qInd].

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if len(args) == 1:
            if args[0].size == self.totalDof:
                # then the input vector has the correct size
                self.q0 = args[0]
            else:
                print('Body {}: error on initial conditions attribution: expected a {}-dimensional vector'.format(self.name,self.totalDof))
        elif len(args) == 2:
            self.q0[args[0]] = args[1]
        else:
            print('{} setInitialConditions: expected 1 or 2 elements.'.format(self.name))
    
    def setVelocityInitialConditions(self,*args):
        '''
        Set the initial conditions on position level
        
        This function has two possible calls
        
        setPositionInitialConditions(q) expects q to be an array with all dofs specified
        
        setPositionInitialConditions(qInd,val) expects qInd to be the index of the dof and val is the initial value attributed to q[qInd].

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if len(args) == 1:
            if args[0].size == self.totalDof:
                # then the input vector has the correct size
                self.u0 = args[0]
            else:
                print('Body {}: error on initial conditions attribution: expected a {}-dimensional vector'.format(self.name,self.totalDof))
        elif len(args) == 2:
            self.u0[args[0]] = args[1]
        else:
            print('{} setInitialConditions: expected 1 or 2 elements.'.format(self.name))
            
            
###############################################################################
############## GROUND #########################################################
############################################################################### 
cdef class ground(body): 
    def __init__(self):        
        super().__init__('Ground',0)
        self.type = 'Ground'
        self.massMatrix = np.array([[]])
        self.addMarker(mbs.marker('O'))
        self.globalDof = []
    
    @property
    def massMatrix(self):
        return np.array(self.massMatrix)

###############################################################################
############## RIGID BODIES ###################################################
###############################################################################  
cdef class rigidBody(body):  
    
    cdef double[:,:] inertiaTensor
    
    def __init__(self,name_):        
        super().__init__(name_,6)
        self.type = 'Rigid body'
        
        self.addMarker(mbs.marker('cog'))
        self.markers[0].setParent(self)
        
        self.massMatrix = np.zeros((6,6))
        
    def setMass(self,new_mass):
        cdef Py_ssize_t i
        self.mass = new_mass
        for i in range(3):
            self.massMatrix[i,i] = new_mass
        
    @property
    def massMatrix(self):
        return np.array(self.massMatrix)
        
    @property
    def inertiaTensor(self):
        return np.array(self.inertiaTensor)
        
        
    def setInertiaTensor(self, tensor):
        mMatrix = np.zeros((6,6))
        if tensor.shape == (3,3):
            self.inertiaTensor = tensor
            mMatrix[:3,:3] = np.diag([self.mass]*3)
            mMatrix[3:,3:] = self.inertiaTensor
            self.massMatrix = mMatrix
        else:
            print('Error in inertia attribution. Need a 3x3 tensor.')
    
    def hVector(self, angDer):
        '''
        Gets the nonlinear angular acceleration forces

        Parameters
        ----------
        angVelo : array of three doubles
            caculated derivatie of the Cardan angles

        Returns
        -------
        h : array of three doubles
            inertial forces vector

        '''
        cdef double cosa = np.cos(angDer[0])
        cdef double sina = np.sin(angDer[0])
        cdef double cosb = np.cos(angDer[1])
        cdef double sinb = np.sin(angDer[1])
        
        omega = np.zeros(3)
        omega[0] = angDer[0] + sinb*angDer[2]
        omega[1] = cosa*angDer[1] - sina*cosb*angDer[2]
        omega[2] = sina*angDer[1] + cosa*cosb*angDer[2]
        
        omtil = hf.skew(omega)
        I = np.array(self.inertiaTensor)
        
        return omtil.dot(I.dot(omega))
            
            
    def info(self):
        print('INFO:')
        print('Printing information on {} ""{}""'.format(self.type,self.name));
        print('Number of dof: {}'.format(self.totalDof))
        print('Inertia tensor: \n{}'.format(np.array(self.inertiaTensor)))
        print('Contains markers:')
        for m in self.markers:
            print('\t{} at {}'.format(m.name,m.position))
        print('-'*80)
            
###############################################################################
############## FLEXIBLE BODIES ################################################
###############################################################################           
cdef class flexibleBody(body):
    '''
    Flexible body class
    '''
    
    cdef int dimensionality
    cdef public list elementList
    cdef object material
    cdef double[:] nodalElasticForces
    cdef double[:,:] stiffnessMatrix, dampingMatrix
    cdef bint nonLinear
    
    def __init__(self,name,material):
        '''
        Flex body initialization method

        Parameters
        ----------
        name : STRING
            NAME OF THE BODY.
        material : MATERIAL
            BODY'S MATERIAL.

        Returns
        -------
        None.

        '''
        
        super().__init__(name)
        self.type = 'Flexible body'
        self.material = material
        self.elementList = []
        self.totalDof = 0   # total number of degrees of freedom      
        self.nonLinear = True
        
    @property 
    def mass(self):
        mass = 0.0
        for e in self.elementList:
            mass += e.mass
            
        return mass
    
    @property
    def massMatrix(self):
        return self.assembleMassMatrix()
    
    @property
    def material(self):
        return self.material
    
    @property
    def q(self):
        cdef double [:] q, qel
        cdef Py_ssize_t [:] eleDof_view     #memory view of element's global dof
        cdef Py_ssize_t i
        q = np.zeros(self.totalDof, dtype=np.float64)
        
        for ele in self.elementList:
            qel = ele.q
            eleDof_view = ele.globalDof
            for i in range(qel.shape[0]):
                q[eleDof_view[i]] = qel[i]
        
        return np.array(q)
    
    @property
    def u(self):
        cdef double [:] u, uel
        cdef Py_ssize_t [:] eleDof_view     #memory view of element's global dof
        cdef Py_ssize_t i
        u = np.zeros(self.totalDof, dtype=np.float64)
        
        for ele in self.elementList:
            uel = ele.u
            eleDof_view = ele.globalDof
            for i in range(uel.shape[0]):
                u[eleDof_view[i]] = uel[i]
        
        return np.array(u)
    
    
    @property
    def nonLinear(self):
        return self.nonLinear
    @nonLinear.setter
    def nonLinear(self, str flag):
        '''
        Sets the nonlinear parameter

        Parameters
        ----------
        str flag : string
            Either 'NL' for nonlinear calculations or 'L' for linearized calculations.

        Returns
        -------
        None.

        '''
        nl = {'NL':True,'L':False}
        self.nonLinear = nl[flag]
        
                
    def assembleMassMatrix(self):
        print('Assemblying mass matrix')
         
        M = np.matlib.zeros([self.totalDof, self.totalDof])
        cdef double[:,:] M_view = M
        cdef double[:,:] m
        
        cdef Py_ssize_t i, j, dofi, dofj
        
        for elem in self.elementList:
            m = elem.getMassMatrix()
            for i,dofi in enumerate(elem.globalDof):
                for j,dofj in enumerate(elem.globalDof):
                    M_view[dofi,dofj] += m[i,j]
            
        print('Mass matrix assembly done!')
        return M
    
    
    def assembleElasticForceVector(self,bint veloc = False):
        
        Qe = np.zeros(self.totalDof)
        cdef double[:] Qe_view = Qe
        cdef double[:] Qelem
        cdef Py_ssize_t i, dof
        
        
        if self.nonLinear:
            
            
            for elem in self.elementList:
                if elem.changedStates:
                    Qelem = elem.getNodalElasticForces(veloc)
                else:
                    #Qelem = elem.nodalElasticForces
                    Qelem = elem.getNodalElasticForces(veloc)
                for i,dof in enumerate(elem.globalDof):
                    Qe_view[dof] += Qelem[i]
                    
        else:
            if veloc:
                # damping forces (needs to be scaled with damping factor)
                Qe = np.array(self.stiffnessMatrix).dot(self.u)
            else:
                Qe = np.array(self.stiffnessMatrix).dot(self.q)
                
            
        return Qe.reshape(-1,1)
    
    def getSM(self):
        return np.array(self.stiffnessMatrix)
    def getq(self):
        return np.array(self.q)
    
    def assembleWeightVector(self, g=np.array([0,1])):
        Qg = np.zeros(self.totalDof)
        cdef double [:] Qg_view = Qg
        
        for elem in self.elementList:
            Qelem = elem.getWeightNodalForces(g).reshape(1,-1)
            for i,dof in enumerate(elem.globalDof):
                Qg_view[dof] += Qelem[0,i]
            
        return Qg.reshape(1,-1)
    
    

    def assembleTangentStiffnessMatrix(self):
                         
        print('Assemblying tangent stiffness matrix')
        
        cdef double [:,:] ke
        self.stiffnessMatrix = np.zeros((self.totalDof, self.totalDof))
        
        for elem in self.elementList:
            ke = elem.getTangentStiffnessMatrix()
            for i,dofi in enumerate(elem.globalDof):
                for j,dofj in enumerate(elem.globalDof):
                    self.stiffnessMatrix[dofi,dofj] += ke[i,j]
                    
        
            
        print('Tangent stiffness matrix assembly done!')
        return np.array(self.stiffnessMatrix)
        
    
    def addElement(self, element):
        '''
        Add a list of elements to the flexible body.

        Parameters
        ----------
        element : ELEMENT
            Element object to be added to elementList.

        Returns
        -------
        None.

        '''
        cdef long curGdl = 0
        cdef long nNumber = 0
        cdef nodalDof
        for el in element:
            el.parentBody = self
            for nd in el.nodes:
                nodalDof = len(nd.q)
                nd.globalDof = list(range(curGdl,curGdl+nodalDof))
                curGdl += nodalDof
                nd.marker.name = 'Node_{}'.format(nNumber)
                nNumber += 1
            # we have to remove the influence of the last node, because it also belongs to the next element
            curGdl = el.globalDof[-1]-nodalDof + 1
            nNumber -= 1
        self.elementList.extend(element)
        self.totalDof = el.globalDof[-1] + 1
        self.nodalElasticForces = np.zeros(self.totalDof)
        
        # cleans up repeated nodes
        self.markers = list( dict.fromkeys(self.markers))
        
        # initializes state vectors
        self.q0 = np.zeros(self.totalDof)
        self.u0 = np.zeros(self.totalDof)
        print('Added {0} elements to body ''{1:s}'''.format(len(element),self.name))
        
        
    def plotPositions(self, int pointsPerElement = 5, bint show=False):
        points = np.linspace(-1.,1.,pointsPerElement)
        
        xy = np.empty([0,self.dimensionality])
        
        for ele in self.elementList:
            for i in range(pointsPerElement-1):
                xy = np.vstack([xy,ele.interpolatePosition(points[i],0,0)])
        #add last point
        xy = np.vstack([xy,ele.interpolatePosition(points[-1],0,0)])
                
        if show:
            if self.dimensionality == 2:
                plt.figure()
                plt.plot(xy[:,0],xy[:,1])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
                
            elif self.dimensionality == 3:
                plt.figure()
                ax = plt.axes(projection='3d',proj_type='ortho')
                ax.plot(xy[:,0],xy[:,1],xy[:,2])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.show()
        return xy
    
    def updateDisplacements(self, double [:] z):
        '''
        Updates the positions of the body nodes

        Parameters
        ----------
        z : array like
            New displacements of the nodes.

        Returns
        -------
        None.
com
        '''
        
        cdef Py_ssize_t i
        cdef Py_ssize_t [:] globDof_view
        cdef double[:] newq
        
        for ele in self.elementList:
            # cycle through nodes
            for nd in ele.nodes:
                newq = nd.q
                globDof_view = nd.globalDof
                for i in range(newq.shape[0]):
                    newq[i] = z[globDof_view[i]]
                nd.q = newq
            # finished cycling through nodes
        
    def updateVelocities(self, double [:] zd):
        '''
        Updates the velocities of the body nodes

        Parameters
        ----------
        z : array like
            New velocities of the nodes.

        Returns
        -------
        None.
        '''
        
        cdef Py_ssize_t i
        cdef Py_ssize_t [:] globDof_view
        cdef double[:] newu
        
        for ele in self.elementList:
            # cycle through nodes
            for nd in ele.nodes:
                newu = nd.u
                globDof_view = nd.globalDof
                for i in range(newu.shape[0]):
                    newu[i] = zd[globDof_view[i]]
                nd.u = newu
            # finished cycling through nodes
            

               
            
    
    def totalStrainEnergy(self):
        
        U = 0
        
        for ele in self.elementList:
            U += ele.strainEnergyNorm()
            
        return U
    
    def findElement(self, double[:] point):
        '''
        Finds to which element a point belongs

        Parameters
        ----------
        double[ : ] point
            global coordinates of a point.

        Returns
        -------
        ele : integer
            the number of the element. -1 if no element found

        '''
        if int(len(point)) != self.dimensionality:
            print('Error in findElement: the point has {} coordinates, but the element is {}-dimensional'.format(len(point),self.dimensionality))
        
        cdef Py_ssize_t ele = -1
        cdef Py_ssize_t i
        
        for i in range(len(self.elementList)):
            if self.elementList[i].isPointOnMe(point):
                ele = i
            
        
        return ele


##############################################################################
cdef class flexibleBody3D(flexibleBody):
    '''
    Tri-dimensional flexible body
    '''
    
    
    def __init__(self, name, material):
        super().__init__(name, material)
        
        self.dimensionality = np.int8(3)
        
        print('Created 3D body \'{}\' with material \'{}\''.format(name,material.name))

cdef class flexibleBody2D(flexibleBody):
    '''
    Bi-dimensional flexible body
    '''
    
    
    def __init__(self, name, material):
        super().__init__(name, material)
        
        self.dimensionality = np.int8(2)
        
        print('Created 2D body \'{}\' with material \'{}\''.format(name,material.name))
    
    