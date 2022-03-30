#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:41:18 2022

@author: leonardo
"""
from sympy import symbols, Matrix, cos, sin, diff

a,b,c = symbols(['a','b','c'])

Ra = Matrix([[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]])
Rb = Matrix([[cos(b),0,-sin(b)],[0,1,0],[sin(b),0,cos(b)]])
Rc = Matrix([[cos(c),-sin(c),0],[sin(c),cos(c),0],[0,0,1]])

# rotation in x and y only
Rxy = Rb*Ra

# rotation in x and z
Rxz = Rc*Ra

# rotation in y and z
Ryz = Rc*Rb

# rotation in all axes
Rxyz = Rc*Rb*Ra



# derivatives
dRda = diff(Rxyz,a)