#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 06:45:00 2022

@author: leonardo
"""
from nachbagauer3Dc import node, railANCF3Dquadratic, beamANCF3Dquadratic
from materialsc import linearElasticMaterial
from bodiesc import flexibleBody3D, rigidBody
from profiles import planarProfile
import MultibodySystem as MBS
import numpy as np
from assimulo.solvers import IDA, ODASSL
import matplotlib.pyplot as plt
import helper_funcs as hf


'''
Initialize system
'''
mbs = MBS.MultibodySystem('Trilho com rodeiro')
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
nq2 = []
nel = 4
totalLength = 2 * nel * 0.58
trackWidth = 1.0
for i in range(nel+1):
    nq.append(node([totalLength * i/nel,0.0,-0.5*trackWidth,
                   0.0,0.99968765,0.02499219, #0.0,1.0,0.0,
                   0.0,-0.02499219,0.9968765]))
    nq2.append(node([totalLength * i/nel,0.0,0.5*trackWidth,
                     0.0,0.99968765,-0.02499219,
                     0.0,0.02499219,0.9968765]))


eq = []
eq2 = []
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
    eq2.append(       
        railANCF3Dquadratic(nq2[j],nq2[j+1],
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
rail = flexibleBody3D('Rail L',steel)
rail.addElement(eq)
rail.nonLinear = 'L'
rail.assembleTangentStiffnessMatrix()

rail2 = flexibleBody3D('Rail R',steel)
rail2.addElement(eq2)
rail2.nonLinear = 'L'
rail2.assembleTangentStiffnessMatrix()

class wheelSet(rigidBody):
    def __init__(self,name):
        super().__init__(name)
        wsmass = 100.
        wsInertiaRadial = 1/12*wsmass*(3*0.15**2+trackWidth**2) 
        I = np.diag([1/12*wsmass*trackWidth**2,1/12*wsmass*trackWidth**2,1/2*wsmass*0.15*0.15])
        self.setMass(wsmass)
        self.setInertiaTensor(I)
        self.setPositionInitialConditions(1,0.092902 + 0.41)
        



# wheel = rigidBody('Wheel',)
# wsmass = 100.
# wsInertiaRadial = 1/12*wsmass*(3*0.15**2+trackWidth**2) 
# I = np.diag([1/12*wsmass*trackWidth**2,1/12*wsmass*trackWidth**2,1/2*wsmass*0.15*0.15])
# wheel.setMass(wsmass)
# wheel.setInertiaTensor(I)
# wheel.setPositionInitialConditions(0,0.75)
# wheel.setPositionInitialConditions(1,0.092902 + 0.41)
#wheel.setPositionInitialConditions(2,-0.5*trackWidth)

wheel1 = wheelSet('Wheelset 1')
wheel1.setPositionInitialConditions(0,0.75+1.6)
wheel2 = wheelSet('Wheelset 2')
wheel2.setPositionInitialConditions(0,0.75)

'''
Sleepers
'''
sleeper1 = MBS.force('Sleepers')
sleeper1.connect(rail,rail2)
#sleeper2 = MBS.force('Sleeper 2')
#leeper2.connect(rail2,mbs.ground)

def slpForce(t,p,v,m1,m2):
    leftRail = m1.parent
    rightRail = m2.parent
    
    
    ## Vertical stiffness ##
    # states
    leftDist = p[leftRail.globalDof[1::9]]
    leftVelo = v[leftRail.globalDof[1::9]]
    rightDist = p[rightRail.globalDof[1::9]]
    rightVelo = v[rightRail.globalDof[1::9]]
    
    
    f = np.zeros_like(p)
    f[leftRail.globalDof[1::9]] = 3e6 * leftDist + 3e4 * leftVelo
    f[rightRail.globalDof[1::9]] = 3e6 * rightDist + 3e4 * rightVelo
    
    ## Lateral stiffness
    stiffness = 35e9 * 0.17 * 0.24 / trackWidth
    
    # states
    leftDist = p[leftRail.globalDof[2::9]]
    leftVelo = v[leftRail.globalDof[2::9]]
    rightDist = p[rightRail.globalDof[2::9]]
    rightVelo = v[rightRail.globalDof[2::9]]
    
    ds = leftDist - rightDist
    dv = leftVelo - rightVelo
    
    f[leftRail.globalDof[2::9]] = stiffness * (ds + 0.02*dv) + 1e6 * leftDist
    f[rightRail.globalDof[2::9]] = stiffness * (- ds - 0.02*dv) + 1e6 * rightDist
    
    ## TODO: Rotation stiffness
    leftY = p[leftRail.globalDof[4::9]]
    leftZ = p[leftRail.globalDof[5::9]]
    rightY = p[rightRail.globalDof[4::9]]
    rightZ = p[rightRail.globalDof[5::9]]
    
    
    
    return -f

sleeper1.setForceFunction(slpForce)
#sleeper2.setForceFunction(slpForce)

# Force to move the wheel
forceWheel = MBS.force('Wheel pull force')
forceWheel.connect(wheel1,MBS.ground())
def pullWheelset(t,p,v,m1,m2):
    w = m1.parent
    f = np.zeros_like(p)
    wpos = p[w.globalDof]
    
    if t > 0.1:
        f[w.globalDof[0]] = 10
    
    f[w.globalDof[1:3]] = - wpos[1:3] * 1e3
        
    return f
forceWheel.setForceFunction(pullWheelset)


# truck connection
truckConn = MBS.force('Truck')
truckConn.connect(wheel1,wheel2)
def truckForce(t,p,v,m1,m2):
    f = np.zeros_like(p)
    w1 = m1.parent
    w2 = m2.parent
    
    wpos1 = p[w1.globalDof]
    wpos1[0] -= 1.6 # remove wheelbase
    wpos2 = p[w2.globalDof]
    
    delta = wpos1-wpos2
    
    tforce = - 1e5 * delta
    
    f[w1.globalDof] = tforce
    f[w2.globalDof] = -tforce
    
    return f
truckConn.setForceFunction(truckForce)


'''
Profiles
'''
rProf = planarProfile('rail', convPar=-1)
rProf.setProfilePointsFromFile('./tr68.dat')
rProf.centerProfile()

rail.addProfile(rProf)

'''
Contact
'''

contactL1 = MBS.force('Contact left wheel to rail')
contactL1.connect(rail,wheel1,pt2=np.array([0.0,-0.41,-0.5*trackWidth]))
contactR1 = MBS.force('Contact right wheel to rail')
contactR1.connect(rail2,wheel1,pt2=np.array([0.0,-0.41,0.5*trackWidth]))

contactL2 = MBS.force('Contact left wheel to rail')
contactL2.connect(rail,wheel2,pt2=np.array([0.0,-0.41,-0.5*trackWidth]))
contactR2 = MBS.force('Contact right wheel to rail')
contactR2.connect(rail2,wheel2,pt2=np.array([0.0,-0.41,0.5*trackWidth]))


def cForce(t,p,v,m1,m2):
    railBody = m1.parent
    wheelBody = m2.parent
    
    cardans = p[wheelBody.globalDof[3:]]
    cardans[2] = 0
    Rwheel = hf.cardanRotationMatrix(cardans)
    rhoM2 = Rwheel.dot(m2.position)
    pWheel = p[wheelBody.globalDof[:3]] + rhoM2
    railDof = np.array(railBody.globalDof)
    
    isit = railBody.findElement(pWheel)
        
    
    f = np.zeros_like(p)
    if isit >= 0:
        contactElement = railBody.elementList[isit]
        localXi = contactElement.mapToLocalCoords(pWheel)
        
        pRail = contactElement.interpolatePosition(localXi[0],1,localXi[2])
        cNormal = contactElement.shapeFunctionDerivative(localXi[0],1,localXi[2])[3:6,:].dot(contactElement.qtotal)
        
        gap = (pWheel-pRail).dot(hf.unitaryVector(cNormal)[0])
        if gap < 0:
            
            contactForce = cNormal * gap * 300e6
            # print(t)
            # print(railBody.name,gap)
            
            f[railDof[contactElement.globalDof]] +=  np.dot(contactForce, contactElement.shapeFunctionMatrix(localXi[0],1,localXi[2]))
        
            f[wheelBody.globalDof[:3]] -= contactForce
            f[wheelBody.globalDof[3:]] -= hf.skew(rhoM2).dot(contactForce)
    return f

contactL1.setForceFunction(cForce)
contactR1.setForceFunction(cForce)
contactL2.setForceFunction(cForce)
contactR2.setForceFunction(cForce)
    
    
    


'''
Multibody system
'''
mbs.addBody([rail,rail2,wheel1,wheel2])

mbs.addForce(sleeper1)
#mbs.addForce(sleeper2)
mbs.addForce(contactL1)
mbs.addForce(contactR1)
mbs.addForce(contactL2)
mbs.addForce(contactR2)
mbs.addForce(forceWheel)
mbs.addForce(truckConn)

mbs.setupSystem()

#%%
'''
Solution
'''

problem = mbs.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-6
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e2 # Hz
finalTime = 2.4

#DAE.make_consistent('IDA_YA_YDP_INIT')

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)
q = p[:,:mbs.n_p] + mbs.pos0
v = p[:,mbs.n_p:2*mbs.n_p] + mbs.vel0
lam = p[:,2*mbs.n_p:] + mbs.lam0

'''
Post-processing
'''
#%%
mbs.postProcess(t,p,v)
from helper_funcs import unitaryVector as uv
plt.figure()
nplots = 6
k = 0
for i in np.arange(0, p.shape[0],int(p.shape[0]/nplots)):
    rail.updateDisplacements(rail.simQ[i])
    rail2.updateDisplacements(rail2.simQ[i])
    a = rail.plotPositions(5)
    b = rail2.plotPositions(5)
    k += 1
    plt.plot(a[:,0],a[:,1], label='{:.2f} s'.format(t[i]), color='red', alpha = ( k/(nplots+1)))
    plt.plot(b[:,0],b[:,1], label='{:.2f} s'.format(t[i]), color='blue', alpha = ( k/(nplots+1)))
plt.legend()
plt.xlabel('Comprimento ao longo do trilho / m')
plt.ylabel('Deslocamento vertical / m')
plt.title(mbs.name)



''' VPYTHON visuals '''
import vpython as vp
import convert_stl as stl
def run_animation():
    scene = vp.canvas(width=1600,height=700,background=vp.color.gray(0.7),fov=0.001,
                      forward = vp.vec(1,0,0))
    
    wheels = [wheel1,wheel2]
    wheelReps = []
    
    for i in range(2):
        wheelReps.append(stl.stl_to_triangles('Rodeiro montado.stl'))
        wheelReps[i].pos = vp.vec(*wheels[i].simQ[0,:3])
        wheelReps[i].rotate(angle=np.pi/2,axis=vp.vec(1,0,0))
        wheelReps[i].visible = True
        wheelReps[i].color = vp.color.cyan
    
    rail.updateDisplacements(rail.simQ[0])
    rail2.updateDisplacements(rail2.simQ[0])
    path = []
    for p in rail2.plotPositions():
        path.append(vp.vec(*p))
    c2 = vp.curve(path, color=vp.color.blue, radius = 0.01)
    c1 = vp.curve(path, color=vp.color.green, radius = 0.01)
    crails = [c1,c2]    
        
    axisX = vp.arrow(pos=vp.vec(0,0,0),axis=vp.vec(0.5,0,0), shaftwidth=0.01, color=vp.color.red)
    axisY = vp.arrow(pos=vp.vec(0,0,0),axis=vp.vec(0.0,0.5,0), shaftwidth=0.01, color=vp.color.green)
    axisz = vp.arrow(pos=vp.vec(0,0,0),axis=vp.vec(0.0,0,0.5), shaftwidth=0.01, color=vp.color.blue)    
    
    vp.rate(500)
    for i in range(len(t)):
        scene.title =  't = {} s'.format(t[i])
        for n,r in enumerate([rail,rail2]):
            r.updateDisplacements(r.simQ[i])
            for j,p in enumerate(r.plotPositions(eta=1)):
                crails[n].modify(j,vp.vec(*p))
        for i in range(2):
            wheelReps[i].pos.x = wheels[i].simQ[i,0]
            wheelReps[i].pos.y = wheels[i].simQ[i,1]
            wheelReps[i].pos.z = wheels[i].simQ[i,2]
            wheelReps[i].rotate(angle=wheels[i].simU[i,3]/outFreq, axis=vp.vec(1,0,0))
            wheelReps[i].rotate(angle=wheels[i].simU[i,4]/outFreq, axis=vp.vec(0,1,0))
            wheelReps[i].rotate(angle=wheels[i].simU[i,5]/outFreq, axis=vp.vec(0,0,1))
        
run_animation()