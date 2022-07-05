#!/usr/env python

# TODO:
#  - complete listings of scripts; they also have to be in a separate directory
#  - divide packages more cleanly
#  - what to do with unit tests ?

# binary package:
# Win: python setup.py bdist_wininst
# Linux: python setup.py bdist --format=rpm          
# MacOS: 

from distutils.core import setup
#from setuptools import setup

setup(name='Apetite',
      version='0.9',
      description='TimeML and temporal reasoning utilities',
      author='Pascal Denis & Philippe Muller',
      author_email='muller@irit.fr, pascal.denis@inria.fr',
      url='',
      package_dir={'apetite':'src'},
      packages=['apetite','apetite.pulp','apetite.graph','apetite.optimisation'],
      requires=['pulp'],
      package_data={'apetite.pulp':['pulp.cfg'],
                    'apetite':['Config/*'],
                    },
      #data_files=[],
      scripts=['src/autocompare.py','src/graph_visualize.py','src/split_into_folds.py','src/n_fold_experiment.py','src/check_result_consistency.py','src/create_models.py','src/launch_tests.py'],
     )

