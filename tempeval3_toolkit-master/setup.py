#!/usr/env python

# TODO:
#  - complete listings of scripts; they also have to be in a separate directory
#  - divide packages more cleanly
#  - what to do with unit tests ?

# binary package:
# Win: python setup.py bdist_wininst
# Linux: python setup.py bdist --format=rpm          
# MacOS: 

# from distutils.core import setup
from setuptools import setup,find_packages
packages=find_packages(exclude=['data','easy_checking'])
print(packages)
setup(name='Temporal Awareness',
      version='0.1',
      description='Evaluation of temporal relation classifiers',
      author='Naushad',
      author_email='',
      url='',
      # package_dir={'apetite':'src'},
      # packages=['apetite','apetite.pulp','apetite.graph','apetite.optimisation'],
      packages=find_packages(exclude=['data','easy_checking','evaluation_entities']),
      py_modules=['TE3_evaluation']
      # scripts = ['./TE3_evaluation.py']
      # requires=['pulp'],
      # package_data={'apetite.pulp':['pulp.cfg'],
      #               'apetite':['Config/*'],
      #               },
      # #data_files=[],
      # scripts=['src/autocompare.py','src/graph_visualize.py','src/split_into_folds.py','src/n_fold_experiment.py','src/check_result_consistency.py','src/create_models.py','src/launch_tests.py'],
     )

