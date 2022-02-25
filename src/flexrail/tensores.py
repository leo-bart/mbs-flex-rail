# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:59:30 2021

@author: lbaru
"""

from sympy import Array, Symbol, factor, tensorproduct
from numpy import *

eps = []
epsp = []
D = []
t = []
n = 2

for i in range(n):
    row = []
    for j in range(n):
        row.append(Symbol('eps_{0}{1}'.format(i+1,j+1)))
    eps.append(row)
    
for i in range(n):
    row = []
    for j in range(n):
        row.append(Symbol('epsp_{0}{1}'.format(i+1,j+1)))
    epsp.append(row)
    
for i in range(n):
    row3 = []
    for j in range(n):
        row2 = []
        for k in range(n):
            row1 = []
            for l in range(n):
                row1.append(Symbol('D_{}{}{}{}'.format(i+1,j+1,k+1,l+1)))
            row2.append(row1)
        row3.append(row2)
    D.append(row3)
    
sum1 = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                sum1 += epsp[i][j] * D[i][j][k][l] * eps[k][l]
                
sum2 = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                sum2 += eps[k][l] * D[k][l][i][j] * epsp[i][j]

for i in range(n):
    row = []
    for j in range(n):
        tij = 0
        for k in range(n):
            for l in range(n):
                tij += D[i][j][k][l]*eps[k][l]
        row.append(tij)
    t.append(row)
                