#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 07:01:46 2022

@author: leonardo
"""
import vtk
import convert_stl as stl
import vpython as vp
from nachbagauer3Dc import node, railANCF3Dquadratic, beamANCF3Dquadratic
from materialsc import linearElasticMaterial
from bodiesc import flexibleBody3D, wheelset
from profiles import planarProfile
import MBS.MultibodySystem as mbs
import MBS.BodyConnections.Forces
import MBS.BodyConnections.BodyConnection
import MBS.BodyConnections.Contacts.WheelRailContact
import numpy as np
from assimulo.solvers import IDA
import matplotlib.pyplot as plt

# %% SYSTEM SETUP
'''
Initialize system
'''
system = mbs.MultibodySystem('Trilho com rodeiro')
system.gravity = np.array([0, -9.81, 0], dtype=np.float64)

'''
Material
'''
steel = linearElasticMaterial('Steel', E=207e12,
                              nu=0.3,
                              rho=7.85e6)


'''
Mesh
'''
nq = []
nq2 = []
nel = 1
totalLength = 2 * nel * 0.58
trackWidth = 1.0
for i in range(nel+1):
    nq.append(node([totalLength * i/nel, 0.0, -0.5*trackWidth-0.039,
                   0.0, 0.99968765, 0.02499219,  # 0.0,1.0,0.0,
                   0.0, -0.02499219, 0.9968765]))
    nq2.append(node([totalLength * i/nel, 0.0, 0.5*trackWidth+0.039,
                     0.0, 0.99968765, -0.02499219,
                     0.0, 0.02499219, 0.9968765]))


eq = []
eq2 = []
for j in range(nel):
    eq.append(
        railANCF3Dquadratic(nq[j], nq[j+1],
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
        railANCF3Dquadratic(nq2[j], nq2[j+1],
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
rail = flexibleBody3D('Rail L', steel)
rail.addElement(eq)
rail.nonLinear = 'L'
rail.assembleTangentStiffnessMatrix()

rail2 = flexibleBody3D('Rail R', steel)
rail2.addElement(eq2)
rail2.nonLinear = 'L'
rail2.assembleTangentStiffnessMatrix()


# Rail boundary conditions
t1fix = MBS.BodyConnections.BodyConnection.nodeEncastreToRigidBody(
    "Rail 1 fixed joint",
    rail,
    system.ground,
    np.array(
        [0.0, 0.0, -0.5*trackWidth-0.039]),
    np.array([0.0, 0.0, -0.5*trackWidth-0.039]))


wLprofile = planarProfile('Design 2 profile - VALE', './design2.pro')
wRprofile = planarProfile('Design 2 profile - VALE', './design2.pro')
wheel = wheelset('Wheel',
                 wLprofile, wRprofile,
                 0.938,
                 0.8382/2)
wsmass = 140.
wsInertiaRadial = 1/12*wsmass*(3*0.15**2+trackWidth**2)
I = np.diag([1/12*wsmass*trackWidth**2, 1/12*wsmass *
            trackWidth**2, 1/2*wsmass*0.15*0.15])
wheel.setMass(wsmass)
wheel.setInertiaTensor(I)
wheel.setPositionInitialConditions(0, 0.75)
# wheel.setPositionInitialConditions(0,totalLength/2)
wheel.setPositionInitialConditions(1, 0.8382/2 + 0.194157/2)
# wheel.setPositionInitialConditions(2,-0.5*trackWidth)
# wheel.setPositionInitialConditions(3,0.25)

'''
Sleepers
'''
sleeper1 = MBS.BodyConnections.Forces.force('Sleepers')
sleeper1.connect(rail, rail2)
# sleeper2 = MBS.force('Sleeper 2')
# leeper2.connect(rail2,mbs.ground)


def slpForce(t, p, v, m1, m2):
    leftRail = m1.parent
    rightRail = m2.parent

    # Vertical stiffness
    # states
    leftDist = p[leftRail.globalDof[1::9]]
    leftVelo = v[leftRail.globalDof[1::9]]
    rightDist = p[rightRail.globalDof[1::9]]
    rightVelo = v[rightRail.globalDof[1::9]]

    f = np.zeros_like(p)
    f[leftRail.globalDof[1::9]] = 3e6 * leftDist + 3e4 * leftVelo
    f[rightRail.globalDof[1::9]] = 3e6 * rightDist + 3e4 * rightVelo
    # increased stiffness on rail ends
    f[leftRail.globalDof[1]] += 32 * (3e6 * leftDist[0])
    f[leftRail.globalDof[-8]] += 32 * (3e6 * leftDist[-1])
    f[rightRail.globalDof[1]] += 32 * (3e6 * rightDist[0])
    f[rightRail.globalDof[-8]] += 32 * (3e6 * rightDist[-1])

    # Lateral stiffness
    stiffness = 35e9 * 0.17 * 0.24 / trackWidth

    # states
    leftDist = p[leftRail.globalDof[2::9]]
    leftVelo = v[leftRail.globalDof[2::9]]
    rightDist = p[rightRail.globalDof[2::9]]
    rightVelo = v[rightRail.globalDof[2::9]]

    ds = leftDist - rightDist
    dv = leftVelo - rightVelo

    f[leftRail.globalDof[2::9]] = stiffness * (ds + 0.02*dv) + 1e6 * leftDist
    f[rightRail.globalDof[2::9]] = stiffness * \
        (- ds - 0.02*dv) + 1e6 * rightDist

    # TODO: Rotation stiffness
    # leftY = p[leftRail.globalDof[4::9]]
    # leftZ = p[leftRail.globalDof[5::9]]
    # rightY = p[rightRail.globalDof[4::9]]
    # rightZ = p[rightRail.globalDof[5::9]]

    return -f


def slpGap(t, p, v, m1, m2):
    return np.zeros((1, 3)), np.zeros((1, 3))


sleeper1.setForceFunction(slpForce)
sleeper1.setGapFunction(slpGap)
# sleeper2.setForceFunction(slpForce)

# Force to move the wheel
forceWheel = MBS.BodyConnections.Forces.force('Wheel pull force')
forceWheel.connect(wheel, system.ground)


def pullWheelset(t, p, v, m1, m2):
    w = m1.parent
    f = np.zeros_like(p)
    wpos, wvel = pullWstGap(t, p, v, m1, m2)
    # wvel = v[w.globalDof]

    if t > 0.8:
        f[w.globalDof[0]] = 10

    f[w.globalDof[1]] = - wvel[1] * 1e2
    f[w.globalDof[3]] = - wvel[3] * 1e3

    return f


def pullWstGap(t, p, v, m1, m2):
    wstBody = m1.parent
    wpos = p[wstBody.globalDof]
    wvel = v[wstBody.globalDof]
    return wpos, wvel


forceWheel.setForceFunction(pullWheelset)
forceWheel.setGapFunction(pullWstGap)

'''
Profiles
'''
leftRailMarker = rail.addMarker(MBS.marker.marker('Left rail marker',
                                                  np.array(
                                                      [0.0, 0.189, -0.5*trackWidth-0.039]),
                                                  np.array([[0.0, 0.0, 1.0],
                                                            [0.0, -1.0, 0.0],
                                                            [1.0, 0.0, 0.0]])))
rightRailMarker = rail2.addMarker(MBS.marker.marker('Right rail marker',
                                                    np.array(
                                                        [0.0, 0.189,  0.5*trackWidth+0.039]),
                                                    np.array([[0.0, 0.0, 1.0],
                                                              [0.0, -1.0, 0.0],
                                                              [1.0, 0.0, 0.0]])))


rProfL = planarProfile('Left rail', convPar=-1)
rProfL.setProfilePointsFromFile('./tr68.pro')
rProfR = planarProfile('Right rail', convPar=1)
rProfR.setProfilePointsFromFile('./tr68.pro')

rail.addProfile(rProfL, leftRailMarker)
rail2.addProfile(rProfR, rightRailMarker)

rail.profiles[0].rotatePoints(np.arctan(1./40))
rail2.profiles[0].rotatePoints(-np.arctan(1./40))

'''Contact markers'''
leftWheelMarker = wheel.addMarker(MBS.marker.marker('Left wheel marker',
                                                    np.array(
                                                        [0, -0.8382/2, -0.459]),
                                                    np.array([[0.0, 0.0, 1.0],
                                                              [0.0, -1.0, 0.0],
                                                              [1.0, 0.0, 0.0]])))
rightWheelMarker = wheel.addMarker(MBS.marker.marker('Right wheel marker',
                                                     np.array(
                                                         [0, -0.8382/2, 0.459]),
                                                     np.array([[0.0, 0.0, 1.0],
                                                               [0.0, 1.0, 0.0],
                                                               [-1.0, 0.0, 0.0]])))


wheel.addProfile(planarProfile('Left wheel profile', './design2.pro', convPar=1),
                 leftWheelMarker)
wheel.addProfile(planarProfile('Right wheel profile', './design2.pro', convPar=1),
                 rightWheelMarker)

wheel.profiles[2].rotatePoints(np.pi)
wheel.profiles[2].mirrorVert()
wheel.profiles[3].mirrorVert()


'''
CONTACT

poits pt2 below represent the reference contact marker, i.e., the
reference frame of the wheel profile
'''

contactL = MBS.BodyConnections.Contacts.WheelRailContact.wrContact(
    'Contact left wheel to rail')
contactL.connect(rail.profiles[0], wheel.profiles[2])
contactR = MBS.BodyConnections.Contacts.WheelRailContact.wrContact(
    'Contact right wheel to rail')
contactR.connect(rail2.profiles[0], wheel.profiles[3])


'''
Multibody system setup
'''
system.addBody([rail, rail2, wheel])

system.addForce(sleeper1)
system.addForce(contactL)
system.addForce(contactR)
system.addForce(forceWheel)
# system.addConstraint(t1fix)

system.setupSystem()

# %% SOLUTION
'''
Solution
'''

problem = system.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-5
DAE.maxh = 1e-4
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e2  # Hz
finalTime = .5

# DAE.make_consistent('IDA_YA_YDP_INIT')

t, p, v = DAE.simulate(finalTime, finalTime * outFreq)
q = p[:, :system.n_p]
v = p[:, system.n_p:2*system.n_p]
lam = p[:, 2*system.n_p:]

np.savez('results/railOut', t=t, p=p, v=v, lam=lam)

'''
Post-processing
'''
# %% POST PROCESSING
oFiles = np.load('./results/railOut.npz')
t = oFiles['t']
p = oFiles['p']
v = oFiles['v']
lam = oFiles['lam']

system.postProcess(t, p, v)


def plotRails():
    from helper_funcs import unitaryVector as uv
    plt.figure()
    nplots = 4
    k = 0
    for i in np.arange(0, p.shape[0], int(p.shape[0]/nplots)):
        rail.updateDisplacements(rail.simQ[i])
        rail2.updateDisplacements(rail2.simQ[i])
        a = rail.plotPositions(5)
        b = rail2.plotPositions(5)
        k += 1
        plt.plot(a[:, 0], a[:, 2], label='{:.2f} s'.format(
            t[i]), color='red', alpha=(k/(nplots+1)))
        plt.plot(b[:, 0], b[:, 2], label='{:.2f} s'.format(
            t[i]), color='blue', alpha=(k/(nplots+1)))
    plt.legend()
    plt.xlabel('Comprimento ao longo do trilho / m')
    plt.ylabel('Deslocamento vertical / m')
    plt.title(system.name)

    return a, b


'''
Output wheelset configuration
'''


''' VPYTHON visuals '''


def run_animation(vprate=10):
    scene = vp.canvas(width=1600, height=700, background=vp.color.gray(0.7),
                      forward=vp.vec(1, 0, 0), range=0.5)

    # w1 = vp.cylinder(axis = vp.vec(0,0,0.5*trackWidth), radius = 0.15)
    # w2 = w1.clone(axis = vp.vec(0,0,-0.5*trackWidth))
    # wheelRep = vp.compound([w1,w2])
    wheelRep = stl.stl_to_triangles('Rodeiro_carga.stl')
    wheelRep.pos = vp.vec(*wheel.simQ[0, :3])
    wheelRep.rotate(angle=np.pi/2, axis=vp.vec(1, 0, 0))
    wheelRep.visible = True
    wheelRep.color = vp.color.cyan

    rail.updateDisplacements(rail.simQ[0])
    rail2.updateDisplacements(rail2.simQ[0])
    path = []
    for p in rail2.plotPositions():
        path.append(vp.vec(*p))
    c2 = vp.curve(path, color=vp.color.blue, radius=0.01)
    c1 = vp.curve(path, color=vp.color.green, radius=0.01)
    crails = [c1, c2]

    axisX = vp.arrow(pos=vp.vec(0, 0, 0), axis=vp.vec(
        0.5, 0, 0), shaftwidth=0.01, color=vp.color.red)
    axisY = vp.arrow(pos=vp.vec(0, 0, 0), axis=vp.vec(
        0.0, 0.5, 0), shaftwidth=0.01, color=vp.color.green)
    axisz = vp.arrow(pos=vp.vec(0, 0, 0), axis=vp.vec(
        0.0, 0, 0.5), shaftwidth=0.01, color=vp.color.blue)

    for i in range(len(t)):
        vp.rate(vprate)
        scene.title = 't = {} s'.format(t[i])
        for n, r in enumerate([rail, rail2]):
            r.updateDisplacements(r.simQ[i])
            for j, p in enumerate(r.plotPositions(eta=1)):
                crails[n].modify(j, vp.vec(*p))
        wheelRep.pos.x = wheel.simQ[i, 0]
        wheelRep.pos.y = wheel.simQ[i, 1]
        wheelRep.pos.z = wheel.simQ[i, 2]
        wheelRep.rotate(angle=wheel.simU[i, 3]/outFreq, axis=vp.vec(1, 0, 0))
        wheelRep.rotate(angle=wheel.simU[i, 4]/outFreq, axis=vp.vec(0, 1, 0))
        wheelRep.rotate(angle=wheel.simU[i, 5]/outFreq, axis=vp.vec(0, 0, 1))

# run_animation()


'''
Output to Paraview
'''


def load_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def write_vtk(filename, dataset):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(dataset)
    writer.Write()


def output_paraview():
    wheelCenter = np.zeros(3)

    # Load the STL file
    geometry = load_stl('Rodeiro_carga_2.stl')

    for i in range(len(t)):
        wheelCenter[0] = wheel.simQ[i, 0]  # Simulate falling motion
        wheelCenter[1] = wheel.simQ[i, 2]
        wheelCenter[2] = wheel.simQ[i, 1]

        # Translate the geometry to the new position
        transform = vtk.vtkTransform()
        transform.Translate(wheelCenter)

        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(geometry)
        transform_filter.Update()

        polydata = transform_filter.GetOutput()

        # Add wheel.simQ[i,:] values to the point data
        simQ_values = vtk.vtkDoubleArray()
        simQ_values.SetName("SimQ")
        simQ_values.SetNumberOfComponents(3)
        simQ_values.InsertNextTuple(wheel.simQ[i, 0:3])

        polydata.GetPointData().AddArray(simQ_values)

        write_vtk("./results/wheelset_ani_test/wheelset_anim_%04d.vtp" %
                  i, polydata)
