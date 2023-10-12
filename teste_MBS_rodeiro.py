#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 06:48:46 2022

@author: leonardo
"""
import numpy as np
import MBS.MultibodySystem as mbs
import MBS.BodyConnections.Forces
import MBS.marker
import profiles
from bodiesc import rigidBody
from assimulo.solvers import IDA

'''
Wheelset definition
'''

wst_mass = 650.0
wst_Iy = 70.

wst = rigidBody('Wheelset')
wst.setMass(wst_mass)
wst.setInertiaTensor(np.diag([wst_Iy, wst_Iy, wst_Iy]))

profileMarker = wst.addMarker(MBS.marker.marker('Profile',np.array([0.5,-0.47,0.0])))

'''
Profiles
'''
profile = profiles.planarProfile('wheel','./design2.pro', convPar = 1)
wst.addProfile(profile,profileMarker)

wst.profiles[0].rotatePoints(np.pi)

'''
Forces
'''

f = MBS.BodyConnections.Forces.force('Mola')
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