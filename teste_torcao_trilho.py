#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de verificação para torção dos trilhos

Created on Thu Jun  6 07:31:34 2024

@author: leonardo
"""

import MBS.MultibodySystem as mbs
import MBS.BodyConnections.Forces
import MBS.BodyConnections.BodyConnection
import MBS.BodyConnections.Contacts.WheelRailContact
import MBS.Bodies.flexibleTrack
import MBS.BodyConnections.Forces
import numpy as np
import matplotlib.pyplot as plt
import assimulo

# Set up system
system = mbs.MultibodySystem('Teste contato roda-trilho')
system.gravity = np.array([0.0, 0.0, 9.85])

# Track
trackWidth = 1.0

track = MBS.Bodies.flexibleTrack.flexibleTrack('Via',
                                               system=system,
                                               gauge=trackWidth,
                                               length=2.0,  # unused yet
                                               sleeperDistance=0.58,
                                               nel=4,
                                               linearBehavior=True)


# Force
force = MBS.BodyConnections.Forces.force('Força')


def forceFunction(t, p, v, *args):
    """
    Output force vector applied to the rail

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    f = np.zeros_like(p)

    rail = track.leftRail

    xpos = 0.75
    ypos = -0.53
    zpos = 0.002

    applicationPoint = np.array([xpos, ypos, zpos])
    appliedForce = np.array([0.0,
                             -650.0,
                             13300.])

    element = rail.elementList[rail.findElement(applicationPoint)]

    localXi = element.mapToLocalCoords(applicationPoint)

    extForce = np.dot(appliedForce, element.shapeFunctionMatrix(
        localXi[0], localXi[1], localXi[2]))

    f[element.globalDof] += extForce

    return f


force.setForceFunction(forceFunction)
system.addForce(force)

system.setupSystem()


# Solution
problem = system.generate_problem('ind3')

DAE = assimulo.solvers.IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-3
DAE.linear_solver = 'DENSE'
DAE.maxh = 1e-2
DAE.atol = 1e-4
DAE.maxord = 2
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 5e2  # Hz
finalTime = .2

# DAE.make_consistent('IDA_YA_YDP_INIT')

t, p, v = DAE.simulate(finalTime, finalTime * outFreq)
q = p[:, :system.n_p]
v = p[:, system.n_p:2*system.n_p]
lam = p[:, 2*system.n_p:]
system.postProcess(t, p, v)

# Plot final position
plt.subplot(2, 1, 1)

leftPos = track.leftRail.plotPositions()
rightPos = track.rightRail.plotPositions()

plt.plot(leftPos[:, 0], leftPos[:, 2], label='Left rail')
plt.plot(rightPos[:, 0], rightPos[:, 2], label='Right rail')
plt.legend()

plt.subplot(2, 1, 2)

plt.plot(leftPos[:, 0], leftPos[:, 1], label='Left rail')
plt.plot(rightPos[:, 0], rightPos[:, 1], label='Right rail')
plt.legend()

# Torsion of section
angle = np.zeros(len(t)-1)
eta = np.zeros(2)
for i in range(1, len(t)):
    eta = track.leftRail.simQ[i, -3:] + track.leftRail.getq0()[-3:]
    angle[i-1] = np.arctan2(-eta[1], -eta[0])
plt.figure()
plt.plot(t[1:], angle*180/np.pi)

plt.figure()
for i in np.linspace(-1., 1., 6):
    for j in np.linspace(-1., 1., 100):
        x = track.leftRail.elementList[0].interpolatePosition(0, i, j)
        plt.plot(x[1], x[2], 'x')
