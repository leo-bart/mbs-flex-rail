#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:46:57 2022

Sets profiles to be applied to beams and rails

@author: leonardo
"""


import numpy as np

cdef class profile(object):
    
    cdef str name
    
    def __init__(self,name):
        self.name = name
        print('Profile {} created.'.format(name))
        
cdef class planarProfile(profile):
    
    def __init__(self,name):
        super().__init__(name)