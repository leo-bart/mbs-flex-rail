#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 07:01:46 2022

@author: leonardo
"""
from nachbagauer3Dc import node, railANCF3Dquadratic, beamANCF3Dquadratic
from materialsc import linearElasticMaterial
from bodiesc import flexibleBody3D, rigidBody
import MultibodySystem as MBS
import numpy as np
from assimulo.solvers import IDA, ODASSL
import matplotlib.pyplot as plt

'''
Initialize system
'''
mbs = MBS.MultibodySystem('Trilho com bola na ponta')
mbs.gravity = np.array([0,-9.81,0],dtype=np.float64)

'''
Material
'''
steel = linearElasticMaterial('Steel',E = 207e9,
                              nu = 0.3,
                              rho = 7.85e3)


'''
Mesh
'''
nq = []
nel = 4
totalLength = 2 * nel * 0.58
for i in range(nel+1):
    nq.append(node([totalLength * i/nel,0.0,0.0
                   ,0.0,1.0,0.0,
                   0.0,0.0,1.0]))


eq = []
for j in range(nel):
    eq.append(       
        railANCF3Dquadratic(nq[j],nq[j+1],
                                  0.18575,
                                  6*0.0254,
                                  0.0805,
                                  0.022147,
                                  0.032165,
                                  135.605e-3,
                                  23.815e-3,
                                  78.339e-3,
                                  8652.0e-6)
        )
    
'''
Bodies
'''
rail = flexibleBody3D('Rail',steel)
rail.addElement(eq)
rail.nonLinear = 'L'
rail.assembleTangentStiffnessMatrix()



wheel = rigidBody('Wheel',)
I = (2/5 * 16 * 0.1 ** 3) * np.eye(3)
wheel.setMass(16.0)
wheel.setInertiaTensor(0.1*I)
wheel.setPositionInitialConditions(0,totalLength - 0.25)
wheel.setPositionInitialConditions(1,0.09287)

'''
Sleepers
'''
sleeper1 = MBS.force('Sleeper 1')
sleeper1.connect(rail,mbs.ground)

def slpForce(t,p,v,m1,m2):
    dist = p[rail.globalDof[1::9]]
    velo = p[rail.globalDof[1::9]]
    f = np.zeros_like(p)
    f[rail.globalDof[1::9]] = 3e6 * dist + 1e5 * velo
    
    return -f

sleeper1.setForceFunction(slpForce)

'''
Contact
'''

contact = MBS.force('Contact wheel to rail')
contact.connect(rail,wheel)

def cForce(t,p,v,m1,m2):
    pWheel = p[wheel.globalDof[:3]]
    railDof = np.array(rail.globalDof)
    
    def mapToLocalCoords(ele, point, tol = 1e-5):
        '''
        Maps a global point P into local coordinates

        Parameters
        ----------
        double[ : ] point
            global coordinates of a point that is to be mapped locally.

        Returns
        -------
        localP : array
            local coordinates of the point.

        '''
        
        if not ele.isPointOnMe(point):
            print('Error: specified point is not inside the bounding box of this element')
            return 0
        
        
        # initialize local variables
        maxiter = 20
        
        L = totalLength/nel
        H = 0.18575
        W = 6*0.0254
        
        xi_view = np.zeros(3)
        dxi = xi_view
        p = np.array(point)
        
        for i in range(maxiter):
            xi_view += dxi
                       
            rn = ele.interpolatePosition(xi_view[0],xi_view[1],xi_view[2])
            res = p - rn
            if np.all(np.abs(res)<tol):
                break

            J_view = ele.getJacobian(xi_view[0],xi_view[1],xi_view[2]).reshape(3,-1)
            # scaling factors:
            J_view[0] *= L/2
            J_view[1] *= H/2
            J_view[2] *= W/2
            
            dxi = np.linalg.solve(J_view,res)
        
        
        return xi_view
    
    isit = rail.findElement(pWheel)
    
    f = np.zeros_like(p)
    if isit >= 0:
        contactElement = rail.elementList[isit]
        #localXi = contactElement.mapToLocalCoords(pWheel)
        localXi = mapToLocalCoords(contactElement,pWheel)
        
        pRail = contactElement.interpolatePosition(localXi[0],1,localXi[2])
        
        gap = pWheel-pRail
        
        if gap[1] < 0:
        
            contactForce = np.array([0,1.0,0.0]) * gap[1] * 1200e6
            #print(contactForce)
            
            f[railDof[contactElement.globalDof]] +=  np.dot(contactForce, contactElement.shapeFunctionMatrix(localXi[0],1,localXi[2]))
        
            f[wheel.globalDof[:3]] -= contactForce
    return f

contact.setForceFunction(cForce)
    
    
    


'''
Multibody system
'''
mbs.addBody([rail,wheel])

fix = MBS.nodeEncastreToRigidBody('Encastre', rail, mbs.ground, rail.elementList[0].nodes[0].marker, mbs.ground.markers[0])
fix2 = MBS.nodeBallJointToRigidBody('Fix ball', rail, wheel, rail.elementList[-1].nodes[-1].marker, wheel.markers[0])

mbs.addConstraint(fix)
#mbs.addConstraint(fix2)

mbs.addForce(sleeper1)
mbs.addForce(contact)

mbs.setupSystem()
mbs.GT(np.zeros(2*mbs.n_p + mbs.n_la))

'''
Solution
'''

problem = mbs.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-5
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e3 # Hz
finalTime = 1.

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)
q = p[:,:mbs.n_p] + mbs.pos0
v = p[:,mbs.n_p:2*mbs.n_p] + mbs.vel0
lam = p[:,2*mbs.n_p:] + mbs.lam0

'''
Post-processing
'''
mbs.postProcess(t,p,v)
from helper_funcs import unitaryVector as uv
plt.figure()
nplots = 10
k = 0
for i in np.arange(0, p.shape[0],int(p.shape[0]/nplots)):
    rail.updateDisplacements(rail.simQ[i])
    a = rail.plotPositions(5)
    k += 1
    plt.plot(a[:,0],a[:,1], label='{:.2f} s'.format(t[i]), alpha = ( k/(nplots+1)))
    # wheel x
    # wheelCoords = wheel.simQ[i]
    # plt.quiver(*wheelCoords[:2],np.cos(wheelCoords[5]),np.sin(wheelCoords[5]),color = 'blue', 
    #             alpha=k/(nplots+1))
    # plt.quiver(*a[-1,0:2],eq[-1].qtotal[-5],-eq[-1].qtotal[-6],color = 'green', 
    #             alpha=k/(nplots+1))
    # dx = uv(a[-1] - a[-2])[0] 
    # plt.quiver(*a[-2,0:2],dx[0],dx[1],color = 'red',
    #             alpha=k/(nplots+1))
#plt.xlim([0,2.2])
plt.legend()
plt.xlabel('Comprimento ao longo do trilho / m')
plt.ylabel('Deslocamento vertical / m')
plt.title(mbs.name)



'''animation'''
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r')
ln2, = plt.plot([], [], 'o')
plt.title('')

def init():
    ax.set_xlim(0, 1.1*totalLength)
    ax.set_ylim(0.05, 0.15)

def update(frame):
    
    rail.updateDisplacements(rail.simQ[frame])
    a = rail.plotPositions(8)
    xdata = a[:,0]
    ydata = a[:,1] + 0.092875
    ln.set_data(xdata, ydata)
    ln2.set_data(wheel.simQ[frame,0],wheel.simQ[frame,1])
    ax.set_title('t = {:.4f} s'.format(t[frame]))

ani = FuncAnimation(fig, update, frames=[x for x in range(0,len(t),250)],
                    init_func=init, blit=False, save_count=1)
