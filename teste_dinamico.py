# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 22:31:39 2021

@author: lbaru
"""

from nachbagauer import elementLinear, elementQuadratic, node
from materials import linearElasticMaterial
from flexibleBody import flexibleBody
import numpy as np
from matplotlib.pyplot import plot

np.seterr('raise')

'''
TESTE DOS ELEMENTOS COM SIMULAÇÃO DINÂMICA EXPLÍCITA
'''

steel = linearElasticMaterial('Steel',207e3,0.3,7.85e-3)
body = flexibleBody('Bar',steel)
body2 = flexibleBody('Bar quadratic',steel)


n = []
nq = []
nel = 2
totalLength = 2000.
for i in range(nel+1):
    n.append(node(totalLength * i/nel,0.0,0.0,1.0))
    nq.append(node(totalLength * i/nel,0.0,0.0,1.0))


e = []
eq = []
for j in range(nel):
    e.append(elementLinear(n[j],n[j+1],.500,.100))
    eq.append(elementQuadratic(nq[j],nq[j+1],500,100))

body.addElement(e)
body.assembleMassMatrix()
body2.addElement(eq)
body2.assembleMassMatrix()


'''
SOLVER DE EULER
'''

''' RESTRIÇÕES '''
# Constraint matrix 
simBody = body2
conDof = [0,1,2,3]
gdl = simBody.totalDof
Phi = np.zeros([len(conDof),gdl])
# fixed bc
for d in conDof:
    Phi[d,d] = 1
    
    
finalTime = 2
h = 5e-5 # timestep
t = [0]

#LHS matrix
lhs = np.zeros([2*gdl+len(conDof),2*gdl+len(conDof)])
M = simBody.assembleMassMatrix()
In = np.eye(gdl)

lhs[0:gdl,0:gdl] = M
lhs[0:gdl,2*gdl:] = -h*Phi.T 

lhs[gdl:2*gdl,0:gdl] = -h * In
lhs[gdl:2*gdl,gdl:2*gdl] = In

lhs[2*gdl:,gdl:2*gdl] = Phi

lhsInv = np.linalg.inv(lhs)


''' CONDIÇÕES INICIAIS '''
z = [np.zeros([2*gdl+len(conDof)])]


''' FORÇAS EXTERNAS'''
Qa = np.zeros([gdl])
Qa[-3] = 5e5 * 0.5**3


g = [0,-9.81e3*0]



'''Matriz de amortecimento'''
C = 0.002 * simBody.assembleTangentStiffnessMatrix()

outFlag = 0
rhs = np.zeros([2*gdl+len(conDof)])
print('##\nStarting simulation for {} s with timestep {} s\n##'.format(finalTime,h))
while t[-1] < finalTime:
    #Prediction step
    rhs = 0 * rhs
    jac = (-simBody.assembleElasticForceVector().A1 + Qa + simBody.assembleWeightVector(g) - np.dot(C,z[-1][0:gdl]))
    rhs[0:gdl] = np.dot(M,z[-1][0:gdl]) +  jac * h
    rhs[gdl:2*gdl] = z[-1][gdl:2*gdl]
    zPred = np.dot(lhsInv,rhs)
    #update nodal positions
    x = zPred[gdl:2*gdl]
    simBody.updateDisplacements(x)   
    
    
    #Correction step
    jac = 0.5 * (jac + ((-simBody.assembleElasticForceVector().A1 + Qa
                         + simBody.assembleWeightVector(g) 
                         - np.dot(C,z[-1][0:gdl]))))
    rhs[0:gdl] = np.dot(M,z[-1][0:gdl]) +  jac * h
    z.append(np.dot(lhsInv,rhs))
    x = z[-1][gdl:2*gdl]
    simBody.updateDisplacements(x)
    
    
    t.append(t[-1] + h)
    
    outFlag +=1
    
    if outFlag == 1000:
        print('{0:1.6f}'.format(t[-1]))
        #simBody.plotPositions(show=True)
        outFlag = 0
    
    
z = np.asmatrix(z)
q = z[:,:simBody.totalDof]

