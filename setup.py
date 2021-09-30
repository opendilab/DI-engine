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

import os

from setuptools import setup, find_packages
from importlib import import_module

here = os.path.abspath(os.path.dirname(__file__))
meta_module = import_module('ding')
meta = meta_module.__dict__

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    author=meta['__AUTHOR__'],
    author_email=meta['__AUTHOR_EMAIL__'],
    url='https://github.com/opendilab/DI-engine',
    license='Apache License, Version 2.0',
    keywords='Decision AI Engine',
    packages=[
        # framework
        *find_packages(include=('ding', "ding.*")),
        # application
        *find_packages(include=('dizoo'
                                'dizoo.*')),
    ],
    package_data={package_name: ['*.yaml', '*.xml', '*cfg', '*SC2Map']
                  for package_name in find_packages(include=('ding.*'))},
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.10',
        'requests>=2.25.1',
        'six',
        'gym>=0.20.0',  # pypy incompatible
        'torch>=1.3.1,<=1.9.0',  # PyTorch 1.9.0 is available, if some errors, you need to do something like https://github.com/opendilab/DI-engine/discussions/81
        'pyyaml',
        'easydict==1.9',
        'tensorboardX>=2.1,<=2.2',
        'matplotlib',  # pypy incompatible
        'seaborn',
        'yapf==0.29.0',
        'responses~=0.12.1',
        'flask~=1.1.2',
        'lz4',
        'cloudpickle',
        'tabulate',
        'sortedcontainers',
        'click==7.1.2',
        'URLObject~=2.4.3',
        'urllib3>=1.26.5',
        'readerwriterlock',
        'namedlist',
        'opencv-python',  # pypy incompatible
        'enum_tools',
        'scipy',
        'trueskill',
        'h5py',
    ],
    extras_require={
        'test': [
            'pytest==5.1.1',
            'pytest-xdist==1.31.0',
            'pytest-cov==2.8.1',
            'pytest-forked~=1.3.0',
            'pytest-mock~=3.3.1',
            'pytest-rerunfailures~=9.1.1',
            'pytest-timeouts~=1.2.1',
        ],
        'style': [
            'yapf==0.29.0',
            'flake8',
        ],
        'fast': [
            'numpy-stl',
            'numba>=0.53.0',
        ],
        'dist': [
            'redis==3.5.3',
            'redis-py-cluster==2.1.0',
        ],
        'common_env': [
            'ale-py',  # atari
            'box2d-py',
            'cmake>=3.18.4',
            'opencv-python',  # pypy incompatible
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
        'bsuite_env': [
            'bsuite',
        ],
        'minigrid_env': [
            'gym-minigrid',
        ],
        # 'd4rl_env': [
        #     'd4rl @ git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl',
        # ],
        # 'pybulletgym_env': [
        #     'pybulletgym @ git+https://github.com/benelot/pybullet-gym@master#egg=pybulletgym',
        # ],
        'sc2_env': [
            'absl-py>=0.1.0',
            'future',
            'futures; python_version == "2.7"',
            'mpyq',
            'mock',
            'portpicker>=1.2.0',
            'websocket-client',
            'protobuf>=2.6',
            'sk-video',  # pypy incompatible
            'whichcraft',
            'joblib',
        ],
        'k8s': [
            'kubernetes',
        ]
    },
    entry_points={'console_scripts': ['ding=ding.entry.cli:cli']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Science/Research",
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
