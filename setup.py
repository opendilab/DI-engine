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

description = """nerveX: X-Lab Deep Reinforcement Learning Framework"""


setup(
    name='nerveX',
    version='0.1.0b0',
    description='X-Lab Reinforcement Learning Framework',
    long_description=description,
    author='X-Lab',
    license='Apache License, Version 2.0',
    keywords='DRL Framework',
    packages=[
        # framework
        *find_packages(
            include=('nervex', "nervex.*")
        ),
        # application
        *find_packages(
            include=('app_zoo' 'app_zoo.*')
        ),
    ],
    package_data={package_name: ['*.yaml', '*.xml', '*cfg'] for package_name in find_packages(include=('nervex.*'))},
    install_requires=[
        'absl-py>=0.1.0',
        'future',
        'futures; python_version == "2.7"',
        'mock',
        'mpyq',
        'numpy>=1.10',
        'numpy-stl',
        'portpicker>=1.2.0',
        'protobuf>=2.6',
        'requests~=2.24.0',
        'six',
        'sk-video',  # pypy
        'websocket-client',
        'whichcraft',
        'gym',  # pypy
        'atari_py',
#        #'torch>=1.3.1,<1.5',  # 1.3.1+cuda90_cudnn7.6.3_lms
        'joblib',
        'sphinx>=2.2.1',
        'sphinx_rtd_theme',
        'pyyaml',
        'easydict',
        'opencv-python',  # pypy
        'tensorboardX>=2.1',
        'matplotlib',  # pypy
        'yapf==0.29.0',
        'pytest==5.1.1',
        'pytest-xdist==1.31.0',
        'pytest-cov==2.8.1',
        'pytest-forked~=1.3.0',
        'pytest-mock~=3.3.1',
        'pytest-rerunfailures~=9.1.1',
        'pytest-timeouts~=1.2.1',
        'responses~=0.12.1',
        'flask~=1.1.2',
        'lz4',
        'cloudpickle',
        'sumolib',
        'traci',
        'tabulate',
        'torchvision==0.2.1',  # pypy
        'sortedcontainers',
        'click',
        'numba',
        'enum34~=1.1.10',
        'URLObject~=2.4.3',
        'urllib3==1.25.10',
        'redis==3.5.3',
        'redis-py-cluster==2.1.0',
        'cmake>=3.18.4'
    ],
    entry_points={
        'console_scripts': [
            'nervex=nervex.entry.cli:cli'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
