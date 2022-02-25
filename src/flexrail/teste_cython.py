#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the Cython version of the libraries
Created on Tue Nov 23 20:31:44 2021

@author: leonardo
"""

import nachbagauer3Dc as n3c
import nachbagauer3D as n3
import flexibleBody as fb
import materials as mt
import numpy as np

posi = np.array([0.,0.,0.,
                 0.,1.,0,
                 0.,0.,1.])

L = 1
N = 2

nc = []
n = []

for i in range(N+1):
    posi[0] = i/N*L
    nc.append(n3c.node(posi))
    n.append(n3.node(posi.tolist()))

elec = []
ele = []

for i in range(N):
    elec.append(n3c.beamANCF3Dquadratic(nc[i],nc[i+1],1.,1.))
    ele.append(n3.beamANCF3Dquadratic(n[i],n[i+1],1.,1.))


mat = mt.linearElasticMaterial('test', 200, 0.3, 1.)

body = fb.flexibleBody3D('test', mat)
body.addElement(ele)

bodyc = fb.flexibleBody3D('test cython', mat)
bodyc.addElement(elec)

body.plotPositions()