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

here = os.path.abspath(os.path.dirname(__file__))
meta = {}
with open(os.path.join(here, 'ding', '__init__.py'), 'r') as f:
    exec(f.read(), meta)

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    author=meta['__AUTHOR__'],
    license='Apache License, Version 2.0',
    keywords='Decision AI Engine',
    packages=[
        # framework
        *find_packages(include=('ding', "ding.*")),
        # application
        *find_packages(include=('app_zoo'
                                'app_zoo.*')),
    ],
    package_data={package_name: ['*.yaml', '*.xml', '*cfg']
                  for package_name in find_packages(include=('ding.*'))},
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.10',
        'requests~=2.24.0',
        'six',
        'gym>=0.15.3',  # pypy incompatible
        'torch>=1.3.1,<=1.7.1',
        'pyyaml',
        'easydict==1.9',
        'tensorboardX>=2.1,<=2.2',
        'matplotlib',  # pypy incompatible
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
        'opencv-python',  # pypy incompatible
        'enum_tools'
    ],
    extras_require={
        'doc': [
            'sphinx>=2.2.1',
            'sphinx_rtd_theme~=0.4.3',
            'enum_tools',
            'sphinx-toolbox',
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
        'style': [
            'yapf==0.29.0',
            'flake8',
        ],
        'fast': [
            'numpy-stl',
            'numba>=0.53.0',
            'redis==3.5.3',
            'redis-py-cluster==2.1.0',
        ],
        'common_env': [
            'atari_py',
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
    },
    entry_points={'console_scripts': ['ding=ding.entry.cli:cli']},
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
