import os

__TITLE__ = 'DI-engine'
__VERSION__ = 'v0.2.0-b2'
__DESCRIPTION__ = 'X-Lab DRL Framework'
__AUTHOR__ = "X-Lab"
__AUTHOR_EMAIL__ = "niuyazhe@sensetime.com"
__version__ = __VERSION__

enable_hpc_rl = False
enable_linklink = os.environ.get('ENABLE_LINKLINK', 'false').lower() == 'true'
enable_numba = True
