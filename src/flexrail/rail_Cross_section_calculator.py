#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts calculates equivalent I-section beam to match area and moment 
of inertia of a given rail

Created on Wed Feb  9 16:16:34 2022

@author: leonardo
"""

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class crossSection(object):
    def __init__(self,
                 _wa = 0.,
                 _ha = 0.,
                 _wp = 0.,
                 _hp = 0.,
                 _wb = 0.,
                 _hb = 0.):
        self.wa = _wa
        self.ha = _ha
        
        self.wp = _wp
        self.hp = _hp
        
        self.wb = _wb
        self.hb = _hb   
   
    def getArea(self):
        return self.wa*self.ha + self.wp*self.hp + self.wb*self.hb
    
    def totalHeight(self):
        return self.ha + self.hb + self.hp
    
    def centroidHeight(self):
        Qp = self.hp/2 * self.hp*self.wp
        Qa = (self.hp + self.ha/2) * self.ha*self.wa
        Qb = (self.hp + self.ha + self.hb/2) * self.hb*self.wb
        
        return (Qp + Qa + Qb) / self.getArea()
    
    def secondMomentOfInertiaX(self):
        Ip = self.wp * self.hp ** 3 / 12
        Ia = self.wa * self.ha ** 3 / 12
        Ib = self.wb * self.hb ** 3 / 12
        
        Ap = self.wp * self.hp
        Aa = self.wa * self.ha
        Ab = self.wb * self.hb
        
        hC = self.centroidHeight()
        
        rp = hC - self.hp / 2
        ra = hC - (self.hp + self.ha/2)
        rb = hC - (self.hp + self.ha + self.hb/2)
        
        return (Ip + rp*rp*Ap + Ia + ra*ra*Aa + Ib + rb*rb*Ab)
    
    def secondMomentOfInertiaY(self):
        Ip = self.hp * self.wp ** 3 / 12
        Ia = self.ha * self.wa ** 3 / 12
        Ib = self.hb * self.wb ** 3 / 12
               
        return Ip + Ia + Ib
    
    def polarMomentOfInertia(self):
        
        return self.secondMomentOfInertiaY() + self.secondMomentOfInertiaX()
    
    def drawSection(self):
        fig, ax = plt.subplots()
        patim = patches.Rectangle((-self.wp/2,0), self.wp, self.hp, linewidth = 1, facecolor='red')
        
        alma = patches.Rectangle(((-self.wa/2,self.hp)), self.wa, self.ha, linewidth=1, facecolor='green')
        boleto = patches.Rectangle(((-self.wb/2,self.hp+self.ha)), self.wb, self.hb, linewidth=1, facecolor='blue')
        ax.add_patch(patim)
        ax.add_patch(alma)
        ax.add_patch(boleto)
        plt.xlim((-200,200))
        plt.ylim((0,190))
        plt.show()
        
    def printMe(self):
        print('Altura do patim: {:.3f} mm'.format(self.hp))
        print('Largura do patim: {:.3f} mm'.format(self.wp))
        
        print('Altura da alma: {:.3f} mm'.format(self.ha))
        print('Largura da alma: {:.3f} mm'.format(self.wa))
        
        print('Altura do boleto: {:.3f} mm'.format(self.hb))
        print('Largura do boleto: {:.3f} mm'.format(self.wb))
    
    
        
        
# tests    
if __name__ == '__main__':
    # input data
    Ixr = 39.21e6                          # mm⁴ - second moment around x
    Iyr = 6.036e6                          # mm⁴ - second moment around y
    Ipp = 45.51e6                          # mm⁴ - second moment polar
    Ar = 8652                              # mm² - cross sec area
    hC = 85.01                             # mm - centroid height
    
    
    def res(x):
        wp = x[0]
        hp = x[1]
        wb = x[2]
        hb = x[3]
        wa = x[4]
        ha = 185.7 - hp - hb
        
        cSec = crossSection(_wp=wp,_hp=hp,_wa=wa,_ha=ha,_wb=wb,_hb=hb)
        
        residue = np.array([0,0,0,0,0],dtype=np.float64())
        
        residue[0] = (cSec.secondMomentOfInertiaX()/Ixr - 1)
        residue[1] = (cSec.centroidHeight()/hC - 1)
        residue[2] = (cSec.getArea()/Ar - 1)
        residue[3] = (cSec.polarMomentOfInertia()/Ipp - 1)
        residue[4] = (cSec.secondMomentOfInertiaY()/Iyr - 1)
        
        return residue
    
    # first guess
    wp = 155
    hp = 26
    wb = 70
    hb = 37
    wa = 14
    ha = 185.7 - hp - hb
        
    z = root(res,np.array([wp,hp,wb,hb,wa]),method='lm')
    
    cSec = crossSection(_wp=z.x[0],
                        _hp=z.x[1],
                        _wb=z.x[2],
                        _hb=z.x[3],
                        _wa=z.x[4],
                        _ha=185.7-z.x[1]-z.x[3])
    
    errI11 = (cSec.secondMomentOfInertiaX()/Ixr - 1) * 100
    errI22 = (cSec.secondMomentOfInertiaY()/Iyr - 1) * 100
    errHc = (cSec.centroidHeight()/hC - 1) * 100
    errA = (cSec.getArea()/Ar - 1) * 100
    errIp = (cSec.polarMomentOfInertia()/Ipp - 1) * 100
    
    print('Seção encontrada:')
    cSec.printMe()
    
    
    print('\nErros:')
    print('Segundo momento de inércia X: {:.8f} %'.format(errI11))
    print('Segundo momento de inércia Y: {:.8f} %'.format(errI22))
    print('Momento polar de inércia: {:.8f} %'.format(errIp))
    print('Área: {:.8f} %'.format(errA))
    print('Altura do centroide: {:.8f} %'.format(errHc))
    
    cSec.drawSection()
    