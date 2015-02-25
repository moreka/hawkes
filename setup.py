import numpy

__author__ = 'moreka'

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Hello world app',
  ext_modules = cythonize("cascadegenerator.pyx"),
  include_dirs=[numpy.get_include()]
)
