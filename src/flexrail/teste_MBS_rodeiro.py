#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 06:48:46 2022

@author: leonardo
"""
import numpy as np
import MultibodySystem as mbs
from bodiesc import rigidBody
from assimulo.solvers import IDA

'''
Wheelset definition
'''

wst_mass = 650.0
wst_Iy = 70.

wst = rigidBody('Wheelset',3)
wst.setInertiaTensor(np.diag([wst_mass, wst_mass, wst_Iy]))

'''
Forces
'''

f = mbs.force('Mola')
f.connect(wst)

'''
Multibody system definition
'''

system = mbs.MultibodySystem('Single wheelset')

system.addBody(wst)

system.setupSystem()

problem = system.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-5
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e3 # Hz
finalTime = 1.2

problem.res(0,problem.y0,problem.yd0)

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)

q = p[:,:system.n_p]
u = p[:,system.n_p:2*system.n_p]
lam = p[:,2*system.n_p:]