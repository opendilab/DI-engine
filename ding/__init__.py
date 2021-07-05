import os

__TITLE__ = 'DI-engine'
__VERSION__ = 'v0.2.0-b2'
__DESCRIPTION__ = 'Decision AI Engine'
__AUTHOR__ = "OpenDILab Contributors"
__AUTHOR_EMAIL__ = "opendilab@gmail.com"
__version__ = __VERSION__

enable_hpc_rl = False
enable_linklink = os.environ.get('ENABLE_LINKLINK', 'false').lower() == 'true'
enable_numba = True
