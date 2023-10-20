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



system = mbs.MultibodySystem('Single wheelset')

'''
Wheelset definition
'''

wst_mass = 650.0
wst_Iy = 70.

wst = rigidBody('Wheelset')
wst.setMass(wst_mass)
wst.setInertiaTensor(np.diag([wst_Iy, wst_Iy, wst_Iy]))

profileMarker1 = wst.addMarker(MBS.marker.marker('Profile 1',np.array([0.56,-0.47,0.0])))
profileMarker2 = wst.addMarker(MBS.marker.marker('Profile 2',np.array([-0.56,-0.47,0.0])))

railMarker1 = system.ground.addMarker(MBS.marker.marker('Profile rail 1',np.array([0.464,-0.66,0.0])))
railMarker2 = system.ground.addMarker(MBS.marker.marker('Profile rail 2',np.array([-0.464,-0.66,0.0])))

'''
Profiles
'''
wst.addProfile(profiles.planarProfile('Wheel 1','./design2.pro', convPar = 1),profileMarker1)
wst.addProfile(profiles.planarProfile('Wheel 2','./design2.pro', convPar = 1),profileMarker2)


wst.profiles[0].rotatePoints(np.pi)
wst.profiles[1].rotatePoints(np.pi)
wst.profiles[1].mirrorVert()

system.ground.addProfile(profiles.planarProfile('Rail 1','./tr68.pro', convPar = 1),railMarker1)
system.ground.addProfile(profiles.planarProfile('Rail 2','./tr68.pro', convPar = 1),railMarker2)

'''
Forces
'''

f = MBS.BodyConnections.Forces.linearSpring_PtP('Mola',1e4)

f.connect(wst,system.ground)

'''
Multibody system definition
'''

system.addBody(wst)
system.addForce(f)
system.gravity = np.array([0,-9.8,0],dtype=np.float64)

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

system.postProcess(t, p, v)

q = p[:,:system.n_p]
u = p[:,system.n_p:2*system.n_p]
lam = p[:,2*system.n_p:]


import matplotlib.pyplot as plt

ax = plt.axes()

wst.profiles[0].plotMe(ax)
wst.profiles[1].plotMe(ax)
system.ground.profiles[0].plotMe(ax)
system.ground.profiles[1].plotMe(ax)
