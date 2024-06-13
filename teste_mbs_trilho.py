#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 07:01:46 2022

@author: leonardo
"""
import vpython as vp
from bodiesc import wheelset
from profiles import planarProfile
import MBS.MultibodySystem as mbs
import MBS.BodyConnections.Forces
import MBS.BodyConnections.BodyConnection
import MBS.BodyConnections.Contacts.WheelRailContact
import MBS.Bodies.flexibleTrack
import numpy as np
from assimulo.solvers import IDA
import matplotlib.pyplot as plt

# %% SYSTEM SETUP
'''
Initialize system
'''
system = mbs.RailwaySystem('Trilho com rodeiro')
system.gravity = np.array([0., 0., 9.81], dtype=np.float64)

initialVelocity = 0.1  # m/s

'''
Track
'''
trackWidth = 1.0

track = MBS.Bodies.flexibleTrack.flexibleTrack('Via',
                                               system=system,
                                               gauge=trackWidth,
                                               length=2.0,  # unused yet
                                               sleeperDistance=0.58,
                                               nel=5)
track.activeSleepers = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]

rail = track.leftRail
rail2 = track.rightRail

'''
Wheel profiles
'''
wLprofile = planarProfile('Design 2 profile - VALE', './design2.pro', 1)
wRprofile = planarProfile('Design 2 profile - VALE', './design2.pro', -1)
wheelRadius = 0.831 * 0.5
wheel = wheelset('Wheel',
                 wLprofile, wRprofile,
                 b2bDist=0.917,
                 gaugeRadius=wheelRadius)
wsmass = 2700.
wsInertiaRadial = 1/12*wsmass*(3*0.15**2+trackWidth**2)
wsInertiaTensor = np.diag([wsInertiaRadial, 1/2*wsmass*0.15*0.15,
                           wsInertiaRadial])
wheel.setMass(wsmass)
wheel.setInertiaTensor(wsInertiaTensor)
wheel.setPositionInitialConditions(0, 0.75)
wheel.setPositionInitialConditions(2, -0.838261/2)
wheel.setVelocityInitialConditions(0, initialVelocity)
wheel.setVelocityInitialConditions(4, -initialVelocity / wheelRadius)


forceWheel = MBS.BodyConnections.Forces.force('Wheel pull force')
forceWheel.connect(wheel, system.ground)


def pullWheelset(t, p, v, m1, m2):
    """
    Apply forces to wheelset.

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    m1 : TYPE
        DESCRIPTION.
    m2 : TYPE
        DESCRIPTION.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    w = m1.parent
    f = np.zeros_like(p)
    wpos, wvel = pullWstGap(t, p, v, m1, m2)
    # wvel = v[w.globalDof]

    if t > 0.1:
        f[w.globalDof[0]] = 1000

    f[w.globalDof[2]] = - wvel[2] * 1e4
    f[w.globalDof[3]] = - wvel[3] * 1e8

    return f


def pullWstGap(t, p, v, m1, m2):
    """
    Determine gap for pull wheelset forces

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    m1 : TYPE
        DESCRIPTION.
    m2 : TYPE
        DESCRIPTION.

    Returns
    -------
    wpos : TYPE
        DESCRIPTION.
    wvel : TYPE
        DESCRIPTION.

    """
    wstBody = m1.parent
    wpos = p[wstBody.globalDof]
    wvel = v[wstBody.globalDof]
    return wpos, wvel


forceWheel.setForceFunction(pullWheelset)
forceWheel.setGapFunction(pullWstGap)


'''
CONTACT
'''

wheelRailContact = MBS.BodyConnections.Contacts.WheelRailContact.wrContact(
    rail, rail2, wheel, 'Contact test')

'''
Multibody system setup
'''
system.addTrack(track)
system.addBody([wheel])
system.addForce(wheelRailContact)
system.addForce(forceWheel)

system.setupSystem()

# %% SOLUTION
'''
Solution
'''

problem = system.generate_problem('ind3')
wheelRailContact.evaluateForceFunction(0, system.pos0, system.vel0)

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-4
DAE.maxh = 5e-3
DAE.atol = 1e-6
# DAE.maxord = 3
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 5e2  # Hz
finalTime = 2.5

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


def plotRails(nplots=4):
    from helper_funcs import unitaryVector as uv
    plt.figure()
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


def run_animation(vprate=100):
    scene = vp.canvas(width=1600, height=700, background=vp.color.gray(0.7),
                      forward=vp.vec(1, 0, 0),
                      up=vp.vec(0, 0, -1), range=0.5)

    # w1 = vp.cylinder(axis = vp.vec(0,0,0.5*trackWidth), radius = 0.15)
    # w2 = w1.clone(axis = vp.vec(0,0,-0.5*trackWidth))
    # wheelRep = vp.compound([w1,w2])
    wheelRep = stl.stl_to_triangles('Rodeiro_carga.stl')
    wheelRep.pos = vp.vec(*wheel.simQ[0, :3])
    # wheelRep.rotate(angle=np.pi/2, axis=vp.vec(1, 0, 0))
    wheelRep.visible = True
    wheelRep.color = vp.color.cyan

    # rail profile
    rail_points = track.leftRail.profiles[0].points.tolist()
    rail_points.append(rail_points[0])
    rail_points = vp.shapes.points(pos=rail_points, rotate=vp.pi/2)

    rail.updateDisplacements(rail.simQ[0])
    rail2.updateDisplacements(rail2.simQ[0])
    path = []
    for p in rail2.plotPositions():
        path.append(vp.vec(*p))
    # c2 = vp.curve(path, color=vp.color.blue, radius=0.01)
    c2 = vp.extrusion(path=path, shape=rail_points, color=vp.color.blue)
    # c1 = vp.curve(path, color=vp.color.green, radius=0.01)
    c1 = vp.extrusion(path=path, shape=rail_points, color=vp.color.blue)
    crails = [c1, c2]

    axisX = vp.arrow(pos=vp.vec(0, 0, 0), axis=vp.vec(
        0.5, 0, 0), shaftwidth=0.01, color=vp.color.red)
    axisY = vp.arrow(pos=vp.vec(0, 0, 0), axis=vp.vec(
        0.0, 0.5, 0), shaftwidth=0.01, color=vp.color.green)
    axisz = vp.arrow(pos=vp.vec(0, 0, 0), axis=vp.vec(
        0.0, 0, 0.5), shaftwidth=0.01, color=vp.color.blue)

    outFreq = 100
    vp.rate(vprate)
    for i in range(0, len(t), 20):
        scene.title = 't = {} s'.format(t[i])
        for n, r in enumerate([rail, rail2]):
            r.updateDisplacements(r.simQ[i])
            path = [vp.vec(p[0], p[1], p[2])
                    for p in r.plotPositions().tolist()]
            crails[n] = vp.extrusion(
                path=path, shape=rail_points, color=vp.color.blue)
        wheelRep.pos.x = wheel.simQ[i, 0]
        wheelRep.pos.y = wheel.simQ[i, 1]
        wheelRep.pos.z = wheel.simQ[i, 2]
        wheelRep.rotate(angle=wheel.simU[i, 3]/outFreq, axis=vp.vec(1, 0, 0))
        wheelRep.rotate(angle=wheel.simU[i, 4]/outFreq, axis=vp.vec(0, 1, 0))
        wheelRep.rotate(angle=wheel.simU[i, 5]/outFreq, axis=vp.vec(0, 0, 1))

# run_animation()


# '''
# Output to Paraview
# '''


# def load_stl(file_path):
#     reader = vtk.vtkSTLReader()
#     reader.SetFileName(file_path)
#     reader.Update()
#     return reader.GetOutput()


# def write_vtk(filename, dataset):
#     writer = vtk.vtkXMLPolyDataWriter()
#     writer.SetFileName(filename)
#     writer.SetInputData(dataset)
#     writer.Write()


# def output_paraview():
#     wheelCenter = np.zeros(3)

#     # Load the STL file
#     geometry = load_stl('Rodeiro_carga_2.stl')

#     for i in range(len(t)):
#         wheelCenter[0] = wheel.simQ[i, 0]  # Simulate falling motion
#         wheelCenter[1] = wheel.simQ[i, 2]
#         wheelCenter[2] = wheel.simQ[i, 1]

#         # Translate the geometry to the new position
#         transform = vtk.vtkTransform()
#         transform.Translate(wheelCenter)

#         transform_filter = vtk.vtkTransformFilter()
#         transform_filter.SetTransform(transform)
#         transform_filter.SetInputData(geometry)
#         transform_filter.Update()

#         polydata = transform_filter.GetOutput()

#         # Add wheel.simQ[i,:] values to the point data
#         simQ_values = vtk.vtkDoubleArray()
#         simQ_values.SetName("SimQ")
#         simQ_values.SetNumberOfComponents(3)
#         simQ_values.InsertNextTuple(wheel.simQ[i, 0:3])

#         polydata.GetPointData().AddArray(simQ_values)

#         write_vtk("./results/wheelset_ani_test/wheelset_anim_%04d.vtp" %
#                   i, polydata)
