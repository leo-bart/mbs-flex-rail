#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de MBS com uso de um pêndulo

Created on Sat Mar 19 07:49:30 2022

@author: leonardo
"""

import numpy as np
import MultibodySystem as MBS
import matplotlib.pyplot as plt
import copy
from bodiesc import rigidBody
from assimulo.solvers import IDA

mbs = MBS.MultibodySystem('Pêndulo')
mbs.gravity = np.array([0.0,-9.85,0.0])

massa = 1.
compr = 0.75
Iz = 0.05

'''
Pêndulo
'''
pend = rigidBody('Pêndulo A')


pend.setMass(massa)
pend.setInertiaTensor(np.diag([1.,1.,Iz]))
pend.setPositionInitialConditions(0,compr/2)

pend.addMarker(MBS.marker('PtA',
               position = np.array([-compr/2,0.,0.])))
pend.addMarker(MBS.marker('PtB',
               position = np.array([compr/2,0.,0.])))


pend2 = rigidBody('Pêndulo B')
pend2.setMass(2*massa)
pend2.setInertiaTensor(np.diag([1.,1.,Iz]))
pend2.setPositionInitialConditions(0,3*compr/2)

pend2.addMarker(MBS.marker('PtA',
               position = np.array([-compr/2,0.,0.])))
pend2.addMarker(MBS.marker('PtB',
               position = np.array([compr/2,0.,0.])))

mbs.addBody(pend)
mbs.addBody(pend2)

'''
Pino
'''
pino = MBS.hingeJoint('Pino', pend.markers[1], mbs.ground.markers[0])
mbs.addConstraint(pino)

solda = MBS.hingeJoint('Solda', pend.markers[2], pend2.markers[1])
mbs.addConstraint(solda)

'''
Solution
'''
mbs.setupSystem()

problem = mbs.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-6
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e3 # Hz
finalTime = 1.8

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)

plt.plot(p[:,0],p[:,1])
plt.plot(p[:,6],p[:,7])
