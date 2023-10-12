#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:35:57 2022
Sets profiles to be applied to beams and rails
@author: leonardo
"""

import numpy as np
import matplotlib.pyplot as plt
from bodiesc import rigidBody



class profile(object):
    """Generic profile class."""
    
    def __init__(self,name):
        self.name = name
        self.body = None
        self.refPointCoordinates = np.zeros(2)
        print('Profile {} created.'.format(name))
        
    def setParent(self,body):
        """
        Set the parent body.

        Parameters
        ----------
        body : body
            The body to which the profile belongs.

        Returns
        -------
        None.

        """
        self.parent = body
        

class planarProfile(profile):
    """
    Planar profile class.
    
    Points are given in (x,y) form, aligned to the reference marker. This means
    that the z axis of the reference marker is ignored, therefore the user 
    should set the marker accordingly.
    """

    def __init__(self,name, fileName='', convPar = 1):
        """
        Initialize profile object.

        Parameters
        ----------
        name : string
            Name of the profile.
        fileName : string, optional
            Path to the profile file.
        convPar : integer, optional
            Conexity parameter. Used to make spline interpolations. The default is 1.

        Returns
        -------
        None.

        """
        super().__init__(name)
        self.convPar = convPar
        if fileName != '':
            self.setProfilePointsFromFile(fileName)
        
    def setProfilePointsFromFile(self, fileName, colDelimiter=';'):
        """
        Load data points from a text file (in csv-like format).

        Parameters
        ----------
        fileName : TYPE
            Text file name address.
        colDelimiter : TYPE, optional
            File delimiter. The default is ';'.

        Returns
        -------
        None.

        """
        self.setProfilePoints(np.genfromtxt(fileName, delimiter=colDelimiter))

    
    def setProfilePoints(self,pArray):
        self.points = pArray
        self.convexSubsets = []
        
        pShape = self.points.shape
        
        if pShape[1] > pShape[0]:
            self.points.tranpose()
        
        self.nPoints = self.points.shape[0]
        
    
    def centerProfile(self):
        shift = np.zeros(2)
        shift[0] = 0.5 * (self.points[:,0].max() + self.points[:,0].min())
        shift[1] = self.points[:,1].max()
        
        self.points -= shift
        
    def offsetPoints(self, offset):
        """
        Offsets all points.
        
        Receives the offset vector [dx,dy] and offsets all points with
        this vector. This permanently changes the position of the points
        and is supposed to be used to adjust the initial configuration.
        
        To get current point positions considering the reference marker,
        use function getCurrentPosition().
        
        Returns
        -------
        None.
        
        """
        self.points += offset
        
    def mirror(self, ax = 0):
        self.points[:,ax] *= -1
    
    def mirrorVert(self):
        """
        Mirror points around the vertical axis.

        Returns
        -------
        None.

        """
        self.mirror(0)
    
    def mirrorHoriz(self):
        """
        Mirror points around the horizontal axis.

        Returns
        -------
        None.

        """
        self.mirror(1)
        
    def rotatePoints(self, angleInRad):
        """
        Apply rotation to profile points.
        
        This method rotates all points with respect to the origin using a
        angleInRad angle. Then the profile points are update to the rotated
        values.

        Parameters
        ----------
        angleInRad : double
            The rotation angle.

        Returns
        -------
        None.

        """
        R = np.array([[np.cos(angleInRad),np.sin(angleInRad)],
                      [-np.sin(angleInRad),np.cos(angleInRad)]])
        self.points = self.points.dot(R)
        
    def plotMe(self, ax=None):
        if ax == None:
            ax = plt.axes()
        currentPoints = self.getCurrentPosition()
        ax.plot(currentPoints[:,0], currentPoints[:,1])
        ax.grid(True)
            
    def plotSubsets(self,ax=None):
        self.createConvexSubsets()
        if ax == None:
            ax = plt.axes()
        for cs in self.convexSubsets:
            ax.plot(cs[:,0], cs[:,1])
            ax.grid(True)
        
    def createConvexSubsets(self):
        
        self.convexSubsets = []
        
        rot = self.convPar
        currSet = []
        currSet.append(self.points[0])
        for i in range(1,self.nPoints-1):
            firstPoint = currSet[0]
            currPoint = self.points[i]
            prevPoint = self.points[i-1]
            nextPoint = self.points[i+1]
            npm1np = currPoint - prevPoint
            npnpp1 = nextPoint - currPoint
            n0np = currPoint - firstPoint
            angSign = np.sign(np.cross(npm1np,npnpp1))
            currSet.append(self.points[i])
            if angSign * rot < 0.01:
                self.convexSubsets.append(np.array(currSet))
                currSet = []
                currSet.append(self.points[i])
                
        self.convexSubsets.append(np.array(currSet))
        
        return self.convexSubsets
    
    def setReferenceMarker(self,_marker):
        """
        Sets the reference marker of the profile.
        
        This function gets the reference marker, which idealy belongs to the
        parent body, and sets it to the profile.
        
        The purpose of this function is to be called by the Body.addProfile
        method.
        """
        self.referenceMarker = _marker
        
    def getCurrentPosition(self):
        currentPoints = self.points + self.referenceMarker.position[0:1]
        return currentPoints
    
    
    def find_convex_hull(self,points):
        """
        Find the indices of the points that form the convex hull.

        Parameters
        ----------
        points : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Sort the points lexicographically
        sorted_points = np.lexsort(points.T)
        
        # Create arrays to hold the upper and lower hulls
        lower_hull = [sorted_points[0]]
        upper_hull = [sorted_points[0]]
        
        # Compute the lower and upper hulls
        for point in sorted_points[1:]:
            # Update the lower hull
            while len(lower_hull) > 1 and np.cross(
                points[lower_hull[-1]] - points[lower_hull[-2]],
                points[point] - points[lower_hull[-2]]
            ) <= 0:
                lower_hull.pop()
            lower_hull.append(point)
            
            # Update the upper hull
            while len(upper_hull) > 1 and np.cross(
                points[upper_hull[-1]] - points[upper_hull[-2]],
                points[point] - points[upper_hull[-2]]
            ) >= 0:
                upper_hull.pop()
            upper_hull.append(point)
        
        # Combine the lower and upper hulls
        convex_hull = lower_hull + upper_hull[1:-1][::-1]
        
        return points[convex_hull]
    
    def setConvexHull(self):
        self.convexHull = self.find_convex_hull(self.points)
        return self.convexHull
        

if __name__=='__main__':
    b = rigidBody('Trilho')
    
    w = planarProfile('wheel')
    r = planarProfile('rail', convPar=-1)
    
    b.addProfile(r)
    
    w.setProfilePointsFromFile('./design2.pro')
    r.setProfilePointsFromFile('./tr68.pro')
    
    r.centerProfile()
    r.rotatePoints(np.arctan(1/40))
    w.mirrorHoriz()
    w.offsetPoints(np.array([-0.08,0.0032]))
    
    ax = plt.gca()
    # w.plotMe(ax)
    # r.plotMe(ax)
    ax.axis('equal')
    
    ch = r.setConvexHull()
    
    wr = wheelRailContact()
    wr.setWheel(w)
    wr.setRail(r)
    
    pw,pr,iw,ir,n,g = wr.searchContactPoint()
    
    # ax.plot(*pw,'o')
    # ax.plot(*pr,'o')
    # ax.quiver(pr[0],pr[1],*n)
    
    w.createConvexSubsets()
    
    import gjk
    rail = 0
    wheel = 1
    cSubsets = {rail:None,wheel:None}
    cPoints = {rail:None,wheel:None}
    minDist = np.inf
    
    plt.fill(ch[:,0],ch[:,1],edgecolor='blue')
    
    for rSubset in [ch]:
        for wSubset in w.convexSubsets:
            if rSubset[-1,0] > wSubset[1,0]:
                pass
            pRail,pWheel,n,d = gjk.gjk(rSubset,wSubset,np.array([0.,-1.]))
            plt.plot(wSubset[:,0],wSubset[:,1])
            plt.arrow(pWheel[0],pWheel[1],pRail[0]-pWheel[0],pRail[1]-pWheel[1])
            # print(d)
            if d < minDist:
                minDist = d
                cSubsets[rail] = rSubset
                cSubsets[wheel] = wSubset
                cPoints[rail] = pRail
                cPoints[wheel] = pWheel
                cNormal = n
                    
                    
    print(minDist)