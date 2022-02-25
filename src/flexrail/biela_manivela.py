# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:53:16 2021

@author: lbaru
"""

from nachbagauer import elementQuadratic, node
from materials import linearElasticMaterial
from flexibleBody import flexibleBody
import numpy as np
from matplotlib import pyplot as plt

np.seterr('raise')

'''
TESTE DOS ELEMENTOS EM BIELA MANIVELA
'''

steel = linearElasticMaterial('Steel',207e9,0.3,7.85e3)
penA = flexibleBody('Pêndulo A',steel)
penB = flexibleBody('Pêndulo B',steel)

LpenA = 150.0e-3
LpenB = 300.0e-3

altura = 6.0e-3
largura = 6.0e-3

g = [0,-9.810]

nel = 1
nodesA = [0]*(nel*3-1)*4
nodesB = [0]*(nel*3-1)*4
for i in range(nel+1):
    nodesA[i]=node(LpenA * i/nel,0.0,0.0,1.0)
    nodesB[i]=node(LpenA + LpenB * i/nel,0.0,0.0,1.0)
    
elemA = [0]*nel
elemB = [0]*nel
for i in range(nel):
    elemA[i]=elementQuadratic(nodesA[i],nodesA[i+1],altura, largura)
    elemB[i]=elementQuadratic(nodesB[i],nodesB[i+1],altura, largura)
    
    
penA.addElement(elemA)
penB.addElement(elemB)

# número total de graus de liberdade nodais
gdlA = penA.totalDof
gdlB = penB.totalDof
gdls = gdlA + gdlB 


'''RESTRIÇÕES'''
'COM O SOLO'
pin1 = [0,1]
'MOTOR'
motor = [2,3]
'ENTRE BARRAS'
#          x em A            x em B          y em A         y em B
pin2 = [[gdlA-4,gdlA],[gdlA-3, gdlA+1]]
'PISTÂO'
slide = [gdls-3]
'MATRIZ DE RESTRIÇÕES'
Phi = np.zeros([len(pin1)+len(pin2)+len(motor)+len(slide),gdls])

def updatePhi():
    phiLine = 0
    for rdof in pin1:
        Phi[phiLine,rdof] = 1
        phiLine += 1
        
    Phi[phiLine,motor[0]] = 1
    phiLine += 1
    Phi[phiLine,motor[1]] = 1
    phiLine += 1
    
    for rdof in pin2:
        Phi[phiLine,rdof[0]] = 1
        Phi[phiLine,rdof[1]] = -1
        phiLine += 1
        
    for rdof in slide:
        Phi[phiLine,rdof] = 1
        phiLine += 1


'''SOLVER'''
finalTime = 1.0/150
h = 5e-7 # timestep
t = [0]

#LHS matrix
updatePhi()

lhs = np.zeros([2*gdls+Phi.shape[0],2*gdls+Phi.shape[0]])
MA = penA.assembleMassMatrix()
MB = penB.assembleMassMatrix()
In = np.eye(gdls)

lhs[0:gdlA,0:gdlA] = MA
lhs[gdlA:gdls,gdlA:gdls] = MB
lhs[0:gdls,2*gdls:] = -h*Phi.T 

lhs[gdls:2*gdls,0:gdls] = -h * In
lhs[gdls:2*gdls,gdls:2*gdls] = In

lhs[2*gdls:,gdls:2*gdls] = Phi


lhsInv = np.linalg.inv(lhs)


''' CONDIÇÕES INICIAIS '''
z = [np.zeros([2*gdls+Phi.shape[0]])]


outFlag = 0
rhs = np.zeros([2*gdls+Phi.shape[0]])
jac = np.array([0.]*gdls)

CA = 0.00 * penA.assembleTangentStiffnessMatrix()
CB = 0.00 * penB.assembleTangentStiffnessMatrix()

WA = penA.assembleWeightVector(g)
WB = penB.assembleWeightVector(g)

print('##\nStarting simulation for {} s with timestep {} s\n##'.format(finalTime,h))



def getForces(body):
    return (-body.assembleElasticForceVector().A1)

omega = 150

while t[-1] < finalTime:
    t.append(t[-1] + h)
    
    
    #Prediction step
    rhs = 0.0 * rhs
    jac[0:gdlA] = getForces(penA) 
    jac[gdlA:gdls] = getForces(penB) 
    rhs[0:gdlA] = np.dot(MA,z[-1][0:gdlA]) +  jac[0:gdlA] * h
    rhs[gdlA:gdls] = np.dot(MB,z[-1][gdlA:gdls]) + jac[gdlA:gdls] * h
    rhs[gdls:2*gdls] = z[-1][gdls:2*gdls]
    rhs[2*gdls+motor[0]] = -np.sin(omega*t[-1])
    rhs[2*gdls+motor[1]] = np.cos(omega*t[-1]) - 1
    zPred = np.dot(lhsInv,rhs)
    #update nodal positions
    xA = zPred[gdls:gdls+gdlA]
    xB = zPred[gdls+gdlA:2*gdls]
    penA.updateDisplacements(xA)   
    penB.updateDisplacements(xB)
    
    
    #Correction step
    jac[0:gdlA] = 0.5 * ( jac[0:gdlA] + getForces(penA))
    jac[gdlA:gdls] = 0.5 * ( jac[gdlA:gdls] + getForces(penB))
    rhs[0:gdlA] = np.dot(MA,z[-1][0:gdlA]) +  jac[0:gdlA] * h
    rhs[gdlA:gdls] = np.dot(MB,z[-1][gdlA:gdls]) + jac[gdlA:gdls] * h
    z.append(np.dot(lhsInv,rhs))
    xA = z[-1][gdls:gdls+gdlA]
    xB = z[-1][gdls+gdlA:2*gdls]
    penA.updateDisplacements(xA)   
    penB.updateDisplacements(xB)
    
    
    
    
    outFlag +=1
    
    if outFlag == 100:
        print('{0:1.6f}'.format(t[-1]))
        #penA.plotPositions(show=True)
        #penB.plotPositions(show=True)
        outFlag = 0
    
    
z = np.asmatrix(z)

vA = z[:,0:gdlA]
vB = z[:,gdlA:gdls]
qA = z[:,gdls:gdls+gdlA]
qB = z[:,gdls+gdlA:2*gdls]
f = z[:,2*gdls:]