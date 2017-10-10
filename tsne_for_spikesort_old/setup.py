
import numpy as np
#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Build import cythonize
from setuptools import find_packages
from Cython.Distutils import build_ext
from setuptools import setup, Extension



ext_modules = [Extension('spikesorttsne',
                         sources=['./src/sptree.cpp', './src/sp_tsne.cpp', './spikesorttsne.pyx'],
                         include_dirs=[np.get_include(),  'src/'],
                         language='c++')]

ext_modules = cythonize(ext_modules)


setup(name='spikesort_tsne.spikesorttsne',
      version='1.0',
      cmdclass = {"build_ext": build_ext},
      author='George Dimitriadis',
      author_email='g.dimitriadis@ucl.ac.uk',
      url='',
      description='Spikesorting TSNE implementations for python',
      license='Apache License Version 2.0, January 2004',
      packages=find_packages(),
      ext_modules=ext_modules
)