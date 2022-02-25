#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cython compilation setup file

Created on Mon Nov 22 07:35:57 2021

@author: leonardo
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["nachbagauer3Dc.pyx",
                             "flexibleBodyc.pyx",
                             "materialsc.pyx",
                             "teste_estatico_3D_cython.pyx"],
                            language_level=3,
                            language='c++')
)