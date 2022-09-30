#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 07:16:41 2021

@author: leonardo
"""
from sympy import symbols, Matrix, BlockMatrix, eye
from sympy import integrate, diff
from sympy import latex, pprint
from sympy import DotProduct, evalf, lambdify, simplify
import numpy as np


[xi,xi_,eta,zeta,L,H,W] =symbols(['xi','xi_','eta','zeta','L','H','W'])
varList = []
for i in range(27):
    varList.append('q[{}]'.format(i))
q = symbols(varList)
q = Matrix(q)
[d11,d22,d12,d33] = symbols(['d11','d22','d12','d33'])



# elemento quadrático
Sq1 = -2/L**2 * xi * (L/2 - xi)
Sq2 = eta * Sq1
Sq3 = 2/L**2 * xi * (L/2 + xi)
Sq4 = eta * Sq3
Sq5 = - 4/L**2 * (xi - L/2) * (xi + L/2)
Sq6 = eta * Sq5

Sq = Matrix([[Sq1,0,Sq2,0,Sq5,0,Sq6,0,Sq3,0,Sq4,0],
            [0,Sq1,0,Sq2,0,Sq5,0,Sq6,0,Sq3,0,Sq4]])

# elemento linear
S1 = (L/2 - xi)
S2 = eta * S1
S3 = (L/2 + xi)
S4 = eta * S3

Sl = 1/L * Matrix([[S1,0 ,S2,0 ,S3,0 ,S4,0],
                       [0 ,S1,0 ,S2,0 ,S3,0 ,S4]])


# elemento quadrático 3D
#xi_ = 2*xi/L
#first node
S13D = - xi_/2 * (1-xi_)
S23D = eta * S13D
S33D = zeta * S13D
#middle node
S43D = 1 - xi_*xi_
S53D = eta*S43D
S63D = zeta*S43D
#last node
S73D = xi_/2 * (1+xi_)
S83D = eta * S73D
S93D = zeta * S73D

E  = eye(3)

S3D = Matrix([S13D*E,S23D*E,S33D*E,S43D*E,S53D*E,
              S63D*E,
              S73D*E,
              S83D*E,
              S93D*E]).T


S = S3D
Sxi = diff(S,xi)
Seta = diff(S,eta)
Szeta = diff(S,zeta)

Sx_dxi = Sxi[0,:]
Sx_deta = Sxi[1,:]
Sx_dzeta = Sxi[2,:]

# Mass matrix
hf,hw,hh = symbols(['hf','hw','hh'])
wf,ww,wh = symbols(['wf','ww','wh'])
zf = -1 + 2*hf/H
zh = 1-2*hh/H
Mf = integrate(
        integrate(
            integrate(
                S3D.transpose()*S3D,(zeta,-1,zf)),
                    (eta,-1,1))
                        ,(xi_,-1,1))
Mw = integrate(
        integrate(
            integrate(
                S3D.transpose()*S3D,(zeta,zf,zh)),
                    (eta,-1,1))
                        ,(xi_,-1,1))
Mh = integrate(
        integrate(
            integrate(
                S3D.transpose()*S3D,(zeta,zh,1)),
                    (eta,-1,1))
                        ,(xi_,-1,1))


M = Mf + Mw + Mh




