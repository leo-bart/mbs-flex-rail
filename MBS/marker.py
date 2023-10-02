#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:54:22 2023

@author: leonardo
"""

import numpy as np

class marker(object):
    """
    Marker class. Is created aligned to the parent body coordinate system.
    
    Parameters
    ----------
    name_ : string
        Name of the marker
    position : array, optional
        Specifies the position of the array in the parent body
    orientation : array, optional
        Specifies the orientation of the marker with respect to parent body's
        coordinate system. Is a rotation matrix.
    """
    
    def __init__(self,name_,position=np.zeros(3),orientation=np.eye(3)):
        self.name = name_
        self.position = position
        self.orientation = orientation
            
    def setParent(self, parent):
        if parent is not None:
            self.parent = parent
            self.name = parent.name + '/' + self.name
            
    def setPosition(self,posi):
        self.position = posi
        
    def setOrientation(self,orient):
        self.orientation = orient