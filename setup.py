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
with open('README.md', mode='r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    long_description=readme,
    long_description_content_type='text/markdown',
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
    package_data={
        package_name: ['*.yaml', '*.xml', '*cfg', '*SC2Map']
        for package_name in find_packages(include=('ding.*'))
    },
    python_requires=">=3.7",
    install_requires=[
        'setuptools<=66.1.1',
        'yapf==0.29.0',
        'gym==0.25.1',  # pypy incompatible; some environmrnt only support gym==0.22.0
        'gymnasium',
        'torch>=1.1.0',
        'numpy>=1.18.0',
        'DI-treetensor>=0.4.0',
        'DI-toolkit>=0.1.0',
        'trueskill',
        'tensorboardX>=2.2',
        'wandb',
        'matplotlib',
        'easydict==1.9',
        'pyyaml',
        'enum_tools',
        'cloudpickle',
        'hickle',
        'tabulate',
        'click>=7.0.0',
        'requests>=2.25.1',  # interaction
        'flask~=1.1.2',  # interaction
        'responses~=0.12.1',  # interaction
        'URLObject>=2.4.0',  # interaction
        'MarkupSafe==2.0.1',  # interaction, compatibility
        'pynng',  # parallel
        'redis',  # parallel
        'mpire>=2.3.5',  # parallel
    ],
    extras_require={
        'test': [
            'coverage>=5,<=7.0.1',
            'mock>=4.0.3',
            'pytest~=7.0.1',  # required by gym>=0.25.0
            'pytest-cov~=3.0.0',
            'pytest-mock~=3.6.1',
            'pytest-xdist>=1.34.0',
            'pytest-rerunfailures~=10.2',
            'pytest-timeout~=2.0.2',
            'readerwriterlock',
            'pandas',
            'lz4',
            'h5py',
            'scipy',
            'scikit-learn',
            'gym[box2d]==0.25.1',
            'pettingzoo<=1.22.3',
            'opencv-python',  # pypy incompatible
        ],
        'style': [
            'yapf==0.29.0',
            'flake8<=3.9.2',
            'importlib-metadata<5.0.0',  # compatibility
        ],
        'fast': [
            'numpy-stl',
            'numba>=0.53.0',
        ],
        'video':[
            'moviepy',
            'imageio[ffmpeg]',
        ],
        'dist': [
            'redis-py-cluster==2.1.0',
        ],
        'common_env': [
            'ale-py',  # >=0.7.5',  # atari
            'autorom',
            'gym[all]==0.25.1',
            'cmake>=3.18.4',
            'opencv-python',  # pypy incompatible
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
            'minigrid>=2.0.0',
        ],
        # 'd4rl_env': [
        #     'd4rl @ git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl',
        # ],
        # 'pybulletgym_env': [
        #     'pybulletgym @ git+https://github.com/benelot/pybullet-gym@master#egg=pybulletgym',
        # ],
        # 'gym_hybrid_env': [
        #     'gym-hybrid @ git+https://github.com/thomashirtz/gym-hybrid@master#egg=gym-hybrid',
        # ],

        # 'gobigger_env': [
        #     'gobigger @ git+https://github.com/opendilab/GoBigger@main#egg=gobigger',
        # ],
        # 'gym_soccer_env': [
        #     'gym-soccer @ git+https://github.com/LikeJulia/gym-soccer@dev-install-packages#egg=gym-soccer',
        # ],
        'slimevolleygym_env': [
            'slimevolleygym',
        ],
        'smac_env': [
            'pysc2',
        ],
        'k8s': [
            'kubernetes',
        ],
        'envpool': [
            'envpool',
        ],
        # 'dmc2gym': [
        #    'dmc2gym @ git+https://github.com/denisyarats/dmc2gym@master#egg=dmc2gym',
        # ],
        # 'rocket_recycling': [
        #    'rocket_recycling @ git+https://github.com/nighood/rocket-recycling@master#egg=rocket_recycling',
        # ],
        'sokoban': [
            'gym-sokoban',
        ],
        'mario': [
            'gym-super-mario-bros>=7.3.0',
        ],
    },
    entry_points={'console_scripts': ['ding=ding.entry.cli:cli', 'ditask=ding.entry.cli_ditask:cli_ditask']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
