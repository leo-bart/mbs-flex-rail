#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cython version of flexible body

Created on Fri Nov 26 07:35:58 2021

@author: leonardo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########################################
cdef class flexibleBody(object):
    '''
    Flexible body class
    '''
    
    
    cdef list elementList
    
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
        
        self.name = name
        self.material = material
        self.elementList = []
        self.totalDof = 0   # total number of degrees of freedom
        
        
        
        
        
    @property 
    def mass(self):
        mass = 0.0
        for e in self.elementList:
            mass += e.mass
            
        return mass
    
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
    
    
    def assembleElasticForceVector(self,int targetDof = -1):
        
        Qe = np.zeros(self.totalDof)
        cdef double[:] Qe_view = Qe 
        cdef double[:] Qelem
        
        cdef Py_ssize_t i, dof
        
        for elem in self.elementList:
            if elem.changedStates:
                Qelem = elem.getNodalElasticForces()
            else:
                #Qelem = elem.nodalElasticForces
                Qelem = elem.getNodalElasticForces()
            for i,dof in enumerate(elem.globalDof):
                Qe_view[dof] += Qelem[i]
            
        return Qe.reshape(-1,1)
    
    def assembleWeightVector(self, g=np.array([0,1])):
        Qg = np.matlib.zeros(self.totalDof)
        
        for elem in self.elementList:
            Qelem = elem.getWeightNodalForces(g).reshape(1,-1)
            for i,dof in enumerate(elem.globalDof):
                Qg[0,dof] += Qelem[0,i]
            
        return Qg.reshape(1,-1)
    
    

    def assembleTangentStiffnessMatrix(self):
                         
        print('Assemblying tangent stiffness matrix')
         
        Kt = np.matlib.zeros([self.totalDof, self.totalDof])
        
        for elem in self.elementList:
            ke = elem.getTangentStiffnessMatrix()
            for i,dofi in enumerate(elem.globalDof):
                for j,dofj in enumerate(elem.globalDof):
                    Kt[dofi,dofj] += ke[i,j]
            
        print('Tangent stiffness matrix assembly done!')
        return Kt
        
    
    def addElement(self, element):
        '''
        

        Parameters
        ----------
        element : ELEMENT
            Element object to be added to elementList.

        Returns
        -------
        None.

        '''
        # TODO adapt to 3D
        curGdl = 0
        for el in element:
            el.parentBody = self
            for nd in el.nodes:
                nodalDof = len(nd.q)
                nd.globalDof = list(range(curGdl,curGdl+nodalDof))
                curGdl += nodalDof
            curGdl = el.globalDof[-1]-nodalDof + 1
        self.elementList.extend(element)
        self.totalDof = el.globalDof[-1] + 1
        self.nodalElasticForces = np.zeros(self.totalDof)
        
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
        cdef Py_ssize_t ele = -1
        cdef Py_ssize_t i
        
        for i in range(len(self.elementList)):
            if self.elementList[i].isPointOnMe(point):
                ele = i
            
        
        return ele


##############################################################################
class flexibleBody3D(flexibleBody):
    '''
    Tri-dimensional flexible body
    '''
    
    
    def __init__(self, name, material):
        super().__init__(name, material)
        
        self.dimensionality = np.int8(3)
        
        print('Created 3D body \'{}\' with material \'{}\''.format(name,material.name))

class flexibleBody2D(flexibleBody):
    '''
    Bi-dimensional flexible body
    '''
    
    
    def __init__(self, name, material):
        super().__init__(name, material)
        
        self.dimensionality = np.int8(2)
        
        print('Created 2D body \'{}\' with material \'{}\''.format(name,material.name))



if __name__ == '__main__':
    from materials import linearElasticMaterial
    body = flexibleBody3D('teste', linearElasticMaterial('teste', 1, 1, 1))
                