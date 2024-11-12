#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the wheel-rail contact

Created on Wed Jun  5 10:46:05 2024

@author: leonardo
"""

from bodiesc import wheelset
from profiles import planarProfile
import MBS.MultibodySystem as mbs
import MBS.BodyConnections.Forces
import MBS.BodyConnections.BodyConnection
import MBS.BodyConnections.Contacts.WheelRailContact
import MBS.Bodies.flexibleTrack
import numpy as np
import matplotlib.pyplot as plt

# Set up system
system = mbs.MultibodySystem('Teste contato roda-trilho')

# Track
trackWidth = 1.0

track = MBS.Bodies.flexibleTrack.flexibleTrack('Via',
                                               system=system,
                                               gauge=trackWidth,
                                               sleeperDistance=0.58,
                                               nel=4)

# Wheelset
wLprofile = planarProfile('Design 2 profile - VALE', './design2.pro', 1)
wRprofile = planarProfile('Design 2 profile - VALE', './design2.pro', -1)
wheel = wheelset('Wheel',
                 wLprofile, wRprofile,
                 b2bDist=0.917,
                 gaugeRadius=0.831/2)
wsmass = 2700.
wsInertiaRadial = 1/12*wsmass*(3*0.15**2+trackWidth**2)
wsInertiaTensor = np.diag([wsInertiaRadial, 1/2*wsmass*0.15*0.15,
                           wsInertiaRadial])
wheel.setMass(wsmass)
wheel.setInertiaTensor(wsInertiaTensor)
wheel.setPositionInitialConditions(0, 0.75)
wheel.setPositionInitialConditions(2, -0.838260/2)

# Contact
wrc = MBS.BodyConnections.Contacts.WheelRailContact.wrContact(
    track.leftRail, track.rightRail, wheel, 'Contact test')


# Set multibody system
system.addBody([wheel])
system.addForce(wrc)
system.setupSystem()

# Gets data
plt.subplot(1, 2, 1)
a = wrc.evaluateGapFunction(0.0, system.pos0, system.vel0, plotFlag=True)
plt.subplot(1, 2, 2)
b = wrc.evaluateGapFunction(
    0.0, system.pos0, system.vel0, 'right', plotFlag=True)
