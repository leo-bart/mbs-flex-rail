# mbs-flex-rail

This is a Python package to perform Multibody Simulations of railway vehicles seating on flexible rails.

## Instalation

The package runs on Python, but uses Cython support to optimze simulation times. Check that all 
dependencies are met before trying to use it.

### Dependencies

mbs-flex-rail depends on several Python packages. These are provided below with links to Anaconda. 
In case you use another Python package manager, please adjust the instructions accordingly :
1. [numpy](https://anaconda.org/anaconda/numpy) 
2. [matplotlib](https://anaconda.org/conda-forge/matplotlib)
3. [assimulo](https://anaconda.org/conda-forge/assimulo)
4. [cython](https://anaconda.org/conda-forge/cython)

### Base compilation using Cython

Before running any simulations, the Cython code must be compiled to your platform. This can be 
accomplished by running 

    $ python setup.py build_ext --inplace
    
on the installation root directory.
    
The file `setup.py` contains a list of `pyx` files that should be compiled. 
You can tweak it to your needs refering to  [Setuptools documentation](https://setuptools.pypa.io/en/latest/) 
and [Cython documentation](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html).