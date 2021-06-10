# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
meta = {}
with open(os.path.join(here, 'nervex', '__init__.py'), 'r') as f:
    exec(f.read(), meta)

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    author=meta['__AUTHOR__'],
    license='Apache License, Version 2.0',
    keywords='DRL Framework',
    packages=[
        # framework
        *find_packages(include=('nervex', "nervex.*")),
        # application
        *find_packages(include=('app_zoo'
                                'app_zoo.*')),
    ],
    package_data={package_name: ['*.yaml', '*.xml', '*cfg']
                  for package_name in find_packages(include=('nervex.*'))},
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.10',
        'requests~=2.24.0',
        'six',
        'gym>=0.15.3',  # pypy
        #'torch>=1.3.1,<=1.7',  # 1.3.1+cuda90_cudnn7.6.3_lms
        'pyyaml',
        'easydict==1.9',
        'tensorboardX>=2.1',
        'matplotlib',  # pypy
        'yapf==0.29.0',
        'responses~=0.12.1',
        'flask~=1.1.2',
        'lz4',
        'cloudpickle',
        'tabulate',
        'sortedcontainers',
        'click==7.1.2',
        'enum34~=1.1.10',
        'URLObject~=2.4.3',
        'urllib3==1.25.10',
        'readerwriterlock',
        'namedlist',
    ],
    extras_require={
        'doc': [
            'sphinx>=2.2.1',
            'sphinx_rtd_theme',
        ],
        'test': [
            'pytest==5.1.1',
            'pytest-xdist==1.31.0',
            'pytest-cov==2.8.1',
            'pytest-forked~=1.3.0',
            'pytest-mock~=3.3.1',
            'pytest-rerunfailures~=9.1.1',
            'pytest-timeouts~=1.2.1',
        ],
        'fast': [
            'numpy-stl',
            'numba',
            'redis==3.5.3',
            'redis-py-cluster==2.1.0',
        ],
        'common_env': [
            'atari_py',
            'box2d-py',
            'cmake>=3.18.4',
            'opencv-python',  # pypy
        ],
        'sumo_env': [
            'sumolib',
            'traci',
        ],
        'gfootball_env': [
            'gfootball',
            'kaggle-environments',
        ],
        'procgen_env': [
            'procgen',
        ],
        'sc2_env': [
            'absl-py>=0.1.0',
            'future',
            'futures; python_version == "2.7"',
            'mpyq',
            'mock',
            'portpicker>=1.2.0',
            'websocket-client',
            'protobuf>=2.6',
            'sk-video',  # pypy
            'whichcraft',
            'joblib',
        ],
    },
    entry_points={'console_scripts': ['nervex=nervex.entry.cli:cli']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research/Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
