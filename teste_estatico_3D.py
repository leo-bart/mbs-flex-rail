#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 07:24:43 2021

@author: leonardo
"""
from nachbagauer3Dc import node, beamANCF3Dquadratic, railANCF3Dquadratic
from materialsc import linearElasticMaterial
from bodiesc import flexibleBody3D
import numpy as np
from scipy.optimize import fsolve
from time import time
'''
TEST PROGRAM
'''


steel = linearElasticMaterial('Steel', 207e9, 0.3, 7.85e3)
body = flexibleBody3D('Bar', steel)
rail = flexibleBody3D('Rail', steel)


nq = []
nr = []
nel = 5
totalLength = 2.
for i in range(nel+1):
    nq.append(node([totalLength * i/nel, 0.0, 0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0]))
    nr.append(node([totalLength * i/nel, 0.0, 0.0, 0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0]))


eq = []
er = []
for j in range(len(nq)-1):
    eq.append(beamANCF3Dquadratic(nq[j], nq[j+1], 0.18575, 0.0734))
    er.append(railANCF3Dquadratic(nr[j], nr[j+1],
                                  0.18575,
                                  6*0.0254,
                                  0.0805,
                                  0.022147,
                                  0.032165,
                                  135.605e-3,
                                  23.815e-3,
                                  78.339e-3,
                                  8652.e-6))


g = np.matrix([[0.0, -9.81]])

body.addElement(eq)
rail.addElement(er)


# Kt = body.assembleTangentStiffnessMatrix()


def simulate(simBody):
    # Constraint matrix
    conDof = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    gdl = simBody.totalDof
    Phi = np.zeros([len(conDof), gdl])

    # fixed bc
    for d in conDof:
        Phi[d, d] = 1

    # record  = []
    # Qelist=[]
    def f(z):

        ndof = gdl

        x = z[0:ndof]
        lam = z[ndof:]

        simBody.updateDisplacements(np.array(x))

        Qe = simBody.assembleElasticForceVector()
        # Qelist.append(Qe.T)
        Qa = Qe*0
        Qa[-8, 0] = -5.0e5 * 0.5 * 0.5 * 0.5

        goal = [0]*(gdl+4)

        goal[0:ndof] = (-Qe.T + Qa.T - np.dot(Phi.T, lam)).tolist()[0]

        goal[ndof:] = np.dot(Phi, x).tolist()

        # print(f"max(|F|) = {np.max(np.abs(goal)):.8e}")

        # record.append(goal)

        return goal

    z0 = [0.]*(gdl+len(conDof))
    z0[-8] = 5.0e5 * 0.5 * 0.5 * 0.5
    z0[-6] = - z0[-3] * 2
    # z = opt.newton_krylov(f,z0,maxiter=40,f_tol=1e-4,verbose=True)

    ts = time()
    z, info, ier, msg = fsolve(f, z0, full_output=True, col_deriv=True)
    print(msg)
    print('Simulation took {0:1.8g} seconds'.format(time()-ts))

    xy = simBody.plotPositions(show=True)
    tipDisp = xy[-1, :]
    gam = np.pi/2-np.arctan2(z[gdl-5]+1, z[gdl-6])
    U = simBody.totalStrainEnergy()
    print('dx = {0:1.8e} m   | dy = {1:1.8e} m \ntheta = {2:1.8e} rad| Unorm = {3:3.5e} J'.format(
        -z[-18], z[-17], gam, U/1000))

    return z, info, ier, msg


if __name__ == "__main__":
    rail.assembleTangentStiffnessMatrix()
    rail.nonLinear = 'NL'
    z, info, ier, msg = simulate(rail)
