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

from setuptools import setup

description = """nerveX: X-Lab Deep Reinforcement Learning Framework"""


setup(
    name='nerveX',
    version='0.0.2b0',
    description='X-Lab Reinforcement Learning Framework',
    long_description=description,
    author='X-Lab',
    license='Apache License, Version 2.0',
    keywords='DRL Framework',
    packages=[
        'nervex',
        'nervex.model',
        'nervex.envs',
        'nervex.policy',
        'nervex.utils',
        'nervex.torch_utils',
        'nervex.worker',
        'nervex.rl_utils',
        'nervex.data',
        'nervex.system',
        'nervex.league',
        'nervex.entry',
        # application(example)
        'app_zoo.sumo',
        'app_zoo.classic_control',
        'app_zoo.atari',
        'app_zoo.gfootball',
        'app_zoo.alphastar',
        'app_zoo.multiagent_particle',
        'app_zoo.smac',
    ],
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
        'requests',
        'six',
        'sk-video',
        'websocket-client',
        'whichcraft',
        'gym',
        'atari_py',
        'torch>=1.3.1,<1.5',  # 1.3.1+cuda90_cudnn7.6.3_lms
        'joblib',
        'sphinx>=2.2.1',
        'sphinx_rtd_theme',
        'pyyaml',
        'easydict',
        'opencv-python',
        'tensorboardX',
        'matplotlib',
        'yapf==0.29.0',
        'pytest==5.1.1',
        'pytest-xdist==1.31.0',
        'pytest-cov==2.8.1',
        'flask',
        'lz4',
        'cloudpickle',
        'sumolib',
        'traci',
        'tabulate',
        'torchvision==0.2.1',
        'sortedcontainers',
    ],
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
