#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cython compilation setup file

Created on Mon Nov 22 07:35:57 2021

@author: leonardo
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["nachbagauer3Dc.pyx",
                           "materialsc.pyx",
                           "profilesc.pyx",
                           "bodiesc.pyx",
                           "helper_funcs.pyx",
                           "polachContactForces.pyx",
                           "gjkc.pyx"],
                          language_level=3,
                          # language='c++',
                          ),
    include_dirs=[numpy.get_include()]
)
