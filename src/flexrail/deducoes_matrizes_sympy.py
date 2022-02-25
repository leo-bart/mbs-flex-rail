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
from nachbagauer import node, elementLinear
import numpy as np
import flexibleBody, materials


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
xi_ = 2*xi/L
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

#def J(z):
 #   return simplify(Matrix([[Sxxi*z,Sxeta*z],[Syxi*z,Syeta*z]]))

'''
J = Matrix([[Sxxi*q,Sxeta*q],[Syxi*q,Syeta*q]])
F = J
eps = 1/2 * (F.T*F - eye(2))   

Deq = diff(eps,q)

E = Matrix([eps[0,0],eps[1,1],2*eps[0,1]])
[Ey,nu] = symbols(['Ey','nu'])
D0 = Matrix([[Ey,0,0],[0,Ey,0],[0,0,Ey/(2*(1+nu))*0.85]])
D1 = Matrix([[nu*nu,nu,0],[nu,nu*nu,0],[0,0,0]]) * Ey/(1-nu*nu)
T = (D0+D1)*E


# integração completa
U = 1/2 * integrate(integrate( E.T * D0 * E, (xi,-L/2,L/2)),(eta,-H/2,H/2)) + 1/2 * H * integrate(E.T.subs(eta,0) * D1 * E.subs(eta,0), (xi,-L/2,L/2))
#integração reduzida

Qne = diff(U,q)
Qne = Qne[:,0,0,0]


#%% teste
Ey = 207e3
nu = 0.3
L_ = 2000.
H_ = 500
W_ = 100
                
D11 = Ey/(1-nu*nu)
D22 = D11
D12 = D11*nu
D33 = D11*(1-nu)*0.5

n1 = node(0.0,0.0,0.0,1.0)
n2 = node(L_,0.0,0.0,1.0)

ele = elementLinear(n1, n2, H_, W_)
b = flexibleBody.flexibleBody('Body', materials.linearElasticMaterial('mat', Ey, nu, 7.8e-6))
b.addElement([ele])


Qne = Qne.subs([(d11,D11),(d22,D22),(d12,D12),(d33,D33),(L,L_),(H,H_)]) * W_
T = T.subs([(d11,D11),(d22,D22),(d12,D12),(d33,D33),(L,L_),(H,H_)])


n2.q[0,2] = -np.sin(0.1)
n2.q[0,3] = np.cos(0.1)
Q0 = b.assembleElasticForceVector()

q_=n1.qtotal.squeeze().tolist()
q_.extend(n2.qtotal.squeeze().tolist())
q_ = Matrix(q_).T

Qnefunc=lambdify(q,Qne,'numpy')
#T=lambdify(q,T,'numpy')

Qne0 = Qnefunc(q_[0,0],q_[0,1],q_[0,2],q_[0,3],q_[0,4],q_[0,5],q_[0,6],q_[0,7])
#T = T(q_[0,0],q_[0,1],q_[0,2],q_[0,3],q_[0,4],q_[0,5],q_[0,6],q_[0,7])

diff= [(x-y)/x*100 for (x,y) in zip(Q0.A1.tolist(),Qne0)]
print(*diff,sep='\n') # para imprimir verticalmente
#%%
ele.stressTensorByPosition(0, 1)

'''


