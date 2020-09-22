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

description = """SenseStar - StarCraft II Learning Environment

Part1 PySC2:
PySC2 is DeepMind's Python component of the StarCraft II Learning Environment
(SC2LE). It exposes Blizzard Entertainment's StarCraft II Machine Learning API
as a Python RL Environment. This is a collaboration between DeepMind and
Blizzard to develop StarCraft II into a rich environment for RL research. PySC2
provides an interface for RL agents to interact with StarCraft 2, getting
observations and sending actions.

We have published an accompanying blogpost and paper
https://deepmind.com/blog/deepmind-and-blizzard-open-starcraft-ii-ai-research-environment/
which outlines our motivation for using StarCraft II for DeepRL research, and
some initial research results using the environment.

Read the README at https://github.com/deepmind/pysc2 for more information.

Part2 TStarBot1:
Macro-action-based StarCraft-II learning environment.
"""

setup(
    name='nerveX',
    version='0.0.1',
    description='X-lab Reinforcement Learning Framework',
    long_description=description,
    author='X-lab',
    license='Apache License, Version 2.0',
    keywords='RL Framework',
    packages=[
        'nervex',
        'nervex.model',
        'nervex.envs',
        'nervex.computation_graph',
        'nervex.utils',
        'nervex.torch_utils',
        'nervex.worker',
        'nervex.rl_utils',
        'nervex.data',
        'nervex.system',
        'nervex.league',
        'nervex.entry',
    ],
    install_requires=[
        'absl-py>=0.1.0',
        'enum34',
        'future',
        'futures; python_version == "2.7"',
        'mock',
        'mpyq',
        'numpy>=1.10',
        'portpicker>=1.2.0',
        'protobuf>=2.6',
        'requests',
        'six',
        'sk-video',
        'websocket-client',
        'whichcraft',
        'gym',
        'atari_py',
        'torch>=1.3.1',  # 1.3.1+cuda90_cudnn7.6.3_lms
        'joblib',
        'sphinx',
        'sphinx_rtd_theme',
        'pyyaml',
        'easydict',
        'opencv-python',
        'tensorboardX',
        'matplotlib',
        'yapf==0.29.0',
        'pytest==5.1.1',
        'pytest-xdist',
        'flask',
        'lz4',
        'cloudpickle',
        'sumolib',
        'traci',
        'tabulate',
        'torchvision==0.5.0',
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
