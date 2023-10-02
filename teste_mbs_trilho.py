#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 07:01:46 2022

@author: leonardo
"""
from nachbagauer3Dc import node, railANCF3Dquadratic, beamANCF3Dquadratic
from materialsc import linearElasticMaterial
from bodiesc import flexibleBody3D, rigidBody
from profiles import planarProfile
import MBS.MultibodySystem as mbs
import MBS.BodyConnections.Forces
import MBS.BodyConnections.Contacts.Contact
import gjk as gjk
import numpy as np
from assimulo.solvers import IDA, ODASSL
import matplotlib.pyplot as plt
import helper_funcs as hf
from copy import deepcopy

#%% SYSTEM SETUP
'''
Initialize system
'''
mbs = mbs.MultibodySystem('Trilho com rodeiro')
mbs.gravity = np.array([0,-9.81,0],dtype=np.float64)

'''
Material
'''
steel = linearElasticMaterial('Steel',E = 207e12,
                              nu = 0.3,
                              rho = 7.85e3)


'''
Mesh
'''
nq = []
nq2 = []
nel = 1
totalLength = 2 * nel * 0.58
trackWidth = 1.0
for i in range(nel+1):
    nq.append(node([totalLength * i/nel,0.0,-0.5*trackWidth-0.039,
                   0.0,0.99968765,0.02499219, #0.0,1.0,0.0,
                   0.0,-0.02499219,0.9968765]))
    nq2.append(node([totalLength * i/nel,0.0,0.5*trackWidth+0.039,
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



wheel = rigidBody('Wheel',)
wsmass = 140.
wsInertiaRadial = 1/12*wsmass*(3*0.15**2+trackWidth**2) 
I = np.diag([1/12*wsmass*trackWidth**2,1/12*wsmass*trackWidth**2,1/2*wsmass*0.15*0.15])
wheel.setMass(wsmass)
wheel.setInertiaTensor(I)
wheel.setPositionInitialConditions(0,0.75)
#wheel.setPositionInitialConditions(0,totalLength/2)
wheel.setPositionInitialConditions(1,0.8382/2 + 0.19416/2)
#wheel.setPositionInitialConditions(2,-0.5*trackWidth)
# wheel.setPositionInitialConditions(3,0.25)

'''
Sleepers
'''
sleeper1 = MBS.BodyConnections.Forces.force('Sleepers')
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
    # increased stiffness on rail ends
    f[leftRail.globalDof[1]] += 32 * (3e6 * leftDist[0])
    f[leftRail.globalDof[-8]] += 32 * (3e6 * leftDist[-1])
    f[rightRail.globalDof[1]] += 32 * (3e6 * rightDist[0])
    f[rightRail.globalDof[-8]] += 32 * (3e6 * rightDist[-1])
    
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
    # leftY = p[leftRail.globalDof[4::9]]
    # leftZ = p[leftRail.globalDof[5::9]]
    # rightY = p[rightRail.globalDof[4::9]]
    # rightZ = p[rightRail.globalDof[5::9]]
    
    
    
    return -f

sleeper1.setForceFunction(slpForce)
#sleeper2.setForceFunction(slpForce)

# Force to move the wheel
forceWheel = MBS.BodyConnections.Forces.force('Wheel pull force')
forceWheel.connect(wheel,mbs.ground)
def pullWheelset(t,p,v,m1,m2):
    w = m1.parent
    f = np.zeros_like(p)
    # wpos = p[w.globalDof]
    wvel = v[w.globalDof]
     
    if t > 0.8:
        f[w.globalDof[0]] = 10
    
    f[w.globalDof[1]] = - wvel[1] * 1e2
    f[w.globalDof[3]] = - wvel[3] * 1e3
        
    return f
forceWheel.setForceFunction(pullWheelset)

'''
Profiles
'''
rProf = planarProfile('rail', convPar=-1)
rProf.setProfilePointsFromFile('./tr68.pro')
rProf.centerProfile()

rail.addProfile(rProf)
rail2.addProfile(rProf)



wheel.addProfile(planarProfile('wheel','./design2.pro', convPar = 1))
wheel.addProfile(planarProfile('wheel','./design2.pro', convPar = 1))

wheel.profiles[0].rotatePoints(np.pi)
wheel.profiles[1].rotatePoints(np.pi)
wheel.profiles[0].mirrorVert()
wheel.profiles[0].offsetPoints([0.459,-0.8382/2])
wheel.profiles[1].offsetPoints([-0.459,-0.8382/2])

'''
CONTACT

poits pt2 below represent the reference contact marker, i.e., the
reference frame of the wheel profile
'''

contactL = MBS.BodyConnections.Contacts.Contact.contact('Contact left wheel to rail')
contactL.connect(rail,wheel,pt2=np.array([0.0,-0.41,-0.5*trackWidth]))
contactR = MBS.BodyConnections.Contacts.Contact.contact('Contact right wheel to rail')
contactR.connect(rail2,wheel,pt2=np.array([0.0,-0.41,0.5*trackWidth]))


# # def wrContactForce(t,p,v,m1,m2):
# def wrContactForce(t,p,v,*args):
#     """
#     Caculate wheel-rail contact force.
    
#     The wheel profile coordinates is referenced on the wheelset center, i.e.,
#     on the coordinate system placed at the center of the shaft.

#     Parameters
#     ----------
#     t : array
#         Time t.
#     p : array
#         position at time t.
#     v : array
#         velocity at time t.
#     m1 : marker
#         rail profile reference marker.
#     m2 : marker
#         wheel profile reference marker.

#     Returns
#     -------
#     f : array
#         force.

#     """
#     m1 = args[0]
#     m2 = args[1]
    
#     # gets rail and wheelset bodies
#     railBody = m1.parent
#     wstBody = m2.parent
#     # wheelset reference marker position
#     wstP = p[wstBody.globalDof[:3]]
#     # dofs of rail body
#     railDof = np.array(railBody.globalDof)
    
#     # gets wheel Cardan angles and removes rotation around shaft axis
#     cardans = p[wstBody.globalDof[3:]]
#     cardans[2] = 0
#     # gets wheelset rotation matrix
#     Rwst = hf.cardanRotationMatrix(cardans)
#     # rhoM2 is the relative position of the wheel profile, represented on
#     # the glogal reference frame
#     rhoM2 = Rwst.dot(m2.position)
#     # pWheel is the position of the wheel profile reference point on
#     # the global reference frame
#     pWheel = wstP + rhoM2
    
#     # matrix to convert vectors written on the wheelset reference frame
#     # to the profile reference frame
#     wst2prof = np.array([[0,0,1],[0,1,0]])
    
#     # wheelset reference frame position on profile reference frame coordinates
#     wstPp = wst2prof.dot(wstP)
    
    
    
#     # profiles
#     wp = wstBody.profiles[0]
#     rp = railBody.profiles[0]
    
#     # now, I've got to find the contacting element
#     # we will suppose that, for every contacting element, the terminal
#     # nodes must be on opposite sides of the wheelset midplane, i.e.,
#     # extreme bending on contacting elements is not allowed
#     # Then, we check for each element whether the vector joining its
#     # terminal nodes pierces the wheelset midplane.
#     midplaneNormal = Rwst[:,0]
#     for e in railBody.elementList:
#         n1 = e.nodes[0]
#         n2 = e.nodes[-1]
        
#         # projects the distance between the front node and end
#         # node of each element e to the wheel reference point
#         d1 = (n1.qtotal[:3] - pWheel).dot(midplaneNormal)
#         d2 = (n2.qtotal[:3] - pWheel).dot(midplaneNormal)
        
#         # if the signs of d1 and d2 are different, than the element
#         # pierces the midplane and the element e is the contacting element
#         # the loop can be broken
#         if d1*d2 <= 0:
#             break
        
#     # now e is the contact element
#     # it is necessary to find the longitudinal position of the contact plane
#     # we perform a very naive bissection search to find it
    
#     # start with finding the node that is closer to the plane
#     # direction tells the direction of the first search bissection
#     dmin = d1
#     step = 2
#     startXi = -1
#     newXi = startXi
#     while dmin*dmin > 1e-7:
#         # in the following loop, the `newXi` variable outputs the approximate
#         # xi coordinate of the contact point
#         newd = -(pWheel - e.interpolatePosition(newXi+step,0,0)).dot(midplaneNormal)
#         while newd*dmin > 0:
#             step *= 1.2
#             newXi = newXi+step
#             newd = -(pWheel - e.interpolatePosition(newXi+step,0,0)).dot(midplaneNormal)
#         dmin = newd
#         newXi +=step
#         step = -step/2
    
#     railCpointPosi = e.interpolatePosition(newXi,1,0) # note eta = 1
     
#     ########## 
#     # we can now search for the contact point between wheel and rail profiles
#     ##########
#     plot = False # set to TRUE to plot wheel and rail profiles
#     if plot:
#         x = rp.points[:,0] + railCpointPosi[2]
#         y = rp.points[:,1] + railCpointPosi[1]
#         plt.plot(x,y)
        
#         x = wp.points[:,0] + wstP[2]
#         y = wp.points[:,1] + wstP[1]
#         plt.plot(x,y)
    
    
#     # we get the convex subsets of the wheel and rail profiles and
#     # offset them to the global position
    
#     if pWheel[2] < 0:
#         wstFactor = -1
#     else:
#         wstFactor = 1
    
#     wp = wstBody.profiles[0]
#     wpConvSubsets = (wp.createConvexSubsets()).copy() # we make a copy to preserve the original profile
#     A=wst2prof.dot(Rwst.dot(wst2prof.transpose())) # rotation matrix of the wheelset on the profile css
#     for i in range(len(wpConvSubsets)):
#         wpConvSubsets[i][:,0] += wstPp[0]
#         wpConvSubsets[i][:,0] *= wstFactor
#         wpConvSubsets[i][:,1] += wstPp[1]
#         wpConvSubsets[i] = wpConvSubsets[i].dot(A)
#         if plot:
#             plt.plot(wpConvSubsets[i][:,0],wpConvSubsets[i][:,1])
        
#     rp = railBody.profiles[0]
#     # rpConvSubsets = (rp.createConvexSubsets()).copy()
#     # headOffset = 0.01 # offset to artificially increase head height
#     #                   # this prevent degenerate contact conditions when
#     #                   # wheel penetration is large compared to convex subset
#     #                   # total height
#     # for i in range(len(rpConvSubsets)):
#     #     rpConvSubsets[i][:,0] += railCpointPosi[2]
#     #     rpConvSubsets[i][:,1] += railCpointPosi[1]
#     #     rpConvSubsets[i] = np.append(rpConvSubsets[i],[rpConvSubsets[i][-1,:]],axis=0)
#     #     rpConvSubsets[i] = np.append(rpConvSubsets[i],[rpConvSubsets[i][0,:]],axis=0)
#     #     rpConvSubsets[i][-1,1] -= headOffset
#     #     rpConvSubsets[i][-2,1] -= headOffset
        
#     #     if plot:
#     #         plt.plot(rpConvSubsets[i][:,0],rpConvSubsets[i][:,1])
            
    
#     # replace all convex subsets by the original rail profile
#     # this is to try and converge
#     rpConvSubsets = []
#     rpConvSubsets.append(rp.points.copy())
#     rpConvSubsets[0][:,0] += railCpointPosi[2]
#     rpConvSubsets[0][:,1] += railCpointPosi[1]
    
    
#     # find the contact point, if any
#     cSubsets = {"rail":None,"wheel":None}
#     cPoints = {"rail":None,"wheel":None}
#     minDist = np.inf
    
#     for rSubset in rpConvSubsets:
#         for wSubset in wpConvSubsets:
#             if rSubset[-1,0] > wSubset[1,0]:
#                 pRail,pWheel,n,d = gjk.gjk(rSubset,wSubset,np.array([0.,-1.]))
#                 # print(d)
#                 if d < minDist:
#                     minDist = d
#                     cSubsets["rail"] = rSubset
#                     cSubsets["wheel"] = wSubset
#                     cPoints["rail"] = pRail
#                     cPoints["wheel"] = pWheel
#                     cNormal = n
    
#     # plt.fill(cSubsets[rail][:,0],cSubsets["rail"][:,1], edgecolor='blue')
#     # plt.fill(cSubsets[wheel][:,0],cSubsets["wheel"][:,1], edgecolor='orange')
#     # print(minDist)
        
#     f = np.zeros_like(p)
#     if minDist < 0.0:
#         # 2d contact force on the wheel midplane
#         contactForce = 525e8 * minDist * wst2prof.transpose().dot(cNormal)
#         # gets the vector from the wheelset CoG to the contact point
#         # first on profile local coordinates
#         rhoM2star = cPoints["wheel"] - wstPp
#         # then on global coordinates
#         rhoM2star = Rwst.transpose().dot(wst2prof.transpose().dot(rhoM2star))
        
#         f[wstBody.globalDof[:3]] += contactForce
#         if f[-5] < 0:
#             print('Warning: negative contact force {} N'.format(f[-5]))
#         f[wstBody.globalDof[3:]] += hf.skew(rhoM2star).dot(contactForce)
        
#         cPoints["rail"] = Rwst.transpose().dot(wst2prof.transpose().dot(cPoints["rail"]))
#         localXi = e.mapToLocalCoords(cPoints["rail"])
        
#         # normal contact force
#         f[railDof[e.globalDof]] -=  np.dot(
#             contactForce, 
#             e.shapeFunctionMatrix(localXi[0],localXi[1],localXi[2])
#             )
        
#         # tangential contact force
#         # encontrar velocidade dos pontos de contato
#         # aplicar modelo de Kalker
        

        
#     return f

# contactR.setForceFunction(wrContactForce)
# contactL.setForceFunction(wrContactForce)
    

'''
Multibody system setup
'''
mbs.addBody([rail,rail2,wheel])

mbs.addForce(sleeper1)
mbs.addForce(contactL)
mbs.addForce(contactR)
mbs.addForce(forceWheel)

mbs.setupSystem()

#%% SOLUTION
'''
Solution
'''

problem = mbs.generate_problem('ind3')

DAE = IDA(problem)
DAE.report_continuously = True
DAE.inith = 1e-5
DAE.maxh = 1e-4
DAE.num_threads = 12
DAE.suppress_alg = True

outFreq = 10e2 # Hz
finalTime = .002

#DAE.make_consistent('IDA_YA_YDP_INIT')

t,p,v=DAE.simulate(finalTime, finalTime * outFreq)
q = p[:,:mbs.n_p]
v = p[:,mbs.n_p:2*mbs.n_p]
lam = p[:,2*mbs.n_p:]

np.savez('results/railOut',t=t,p=p,v=v,lam=lam)

'''
Post-processing
'''
#%% POST PROCESSING
oFiles = np.load('./results/railOut.npz')
t = oFiles['t']
p = oFiles['p']
v = oFiles['v']
lam = oFiles['lam']

mbs.postProcess(t,p,v)
def plotRails():
    from helper_funcs import unitaryVector as uv
    plt.figure()
    nplots = 4
    k = 0
    for i in np.arange(0, p.shape[0],int(p.shape[0]/nplots)):
        rail.updateDisplacements(rail.simQ[i])
        rail2.updateDisplacements(rail2.simQ[i])
        a = rail.plotPositions(5)
        b = rail2.plotPositions(5)
        k += 1
        plt.plot(a[:,0],a[:,2], label='{:.2f} s'.format(t[i]), color='red', alpha = ( k/(nplots+1)))
        plt.plot(b[:,0],b[:,2], label='{:.2f} s'.format(t[i]), color='blue', alpha = ( k/(nplots+1)))
    plt.legend()
    plt.xlabel('Comprimento ao longo do trilho / m')
    plt.ylabel('Deslocamento vertical / m')
    plt.title(mbs.name)
    
    return a,b

'''
Output wheelset configuration
'''



''' VPYTHON visuals '''
import vpython as vp
import convert_stl as stl
def run_animation(vprate=10):
    scene = vp.canvas(width=1600,height=700,background=vp.color.gray(0.7),
                      forward = vp.vec(1,0,0), range=0.5)
    
    # w1 = vp.cylinder(axis = vp.vec(0,0,0.5*trackWidth), radius = 0.15)
    # w2 = w1.clone(axis = vp.vec(0,0,-0.5*trackWidth))
    # wheelRep = vp.compound([w1,w2])
    wheelRep = stl.stl_to_triangles('Rodeiro_carga.stl')
    wheelRep.pos = vp.vec(*wheel.simQ[0,:3])
    wheelRep.rotate(angle=np.pi/2,axis=vp.vec(1,0,0))
    wheelRep.visible = True
    wheelRep.color = vp.color.cyan
    
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
    
    
    for i in range(len(t)):
        vp.rate(vprate)
        scene.title =  't = {} s'.format(t[i])
        for n,r in enumerate([rail,rail2]):
            r.updateDisplacements(r.simQ[i])
            for j,p in enumerate(r.plotPositions(eta=1)):
                crails[n].modify(j,vp.vec(*p))
        wheelRep.pos.x = wheel.simQ[i,0]
        wheelRep.pos.y = wheel.simQ[i,1]
        wheelRep.pos.z = wheel.simQ[i,2]
        wheelRep.rotate(angle=wheel.simU[i,3]/outFreq, axis=vp.vec(1,0,0))
        wheelRep.rotate(angle=wheel.simU[i,4]/outFreq, axis=vp.vec(0,1,0))
        wheelRep.rotate(angle=wheel.simU[i,5]/outFreq, axis=vp.vec(0,0,1))
        
#run_animation()


'''
Animation on Blender
'''
# import bpy
# import csv
# import math
# def animate_on_blender():
#     # Set up the scene
#     bpy.ops.wm.read_factory_settings(use_empty=True)
#     scene = bpy.context.scene
#     scene.frame_start = 1
#     scene.frame_end = 250  # Adjust as needed
    
#     # Create a sphere (the falling ball)
#     bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 10))
#     sphere = bpy.context.object
    
#     # Import CSV data
#     csv_file = 'path_to_your_data.csv'  # Replace with your CSV file path
#     with open(csv_file, 'r') as csvfile:
#         csvreader = csv.reader(csvfile)
#         next(csvreader)  # Skip header if present
#         for row in csvreader:
#             time, position = map(float, row)
            
#             # Calculate frame number based on time
#             frame = int(time)
            
#             # Set sphere's location at this frame
#             sphere.location.z = position
            
#             # Keyframe the sphere's location
#             sphere.keyframe_insert(data_path="location", frame=frame, index=2)  # index=2 for Z-axis
    
#     # Set up rendering settings
#     scene.render.image_settings.file_format = 'PNG'
#     scene.render.filepath = 'output/frame_#####'  # Adjust the output path as needed
    
#     # Render the animation
#     bpy.ops.render.render(animation=True)
    
#     # Save the blend file (optional)
#     bpy.ops.wm.save_as_mainfile(filepath='falling_ball_animation.blend')
