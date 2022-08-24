import os

__TITLE__ = 'DI-engine'
__VERSION__ = 'v0.4.1'
__DESCRIPTION__ = 'Decision AI Engine'
__AUTHOR__ = "OpenDILab Contributors"
__AUTHOR_EMAIL__ = "opendilab@pjlab.org.cn"
__version__ = __VERSION__

enable_hpc_rl = os.environ.get('ENABLE_DI_HPC', 'false').lower() == 'true'
enable_linklink = os.environ.get('ENABLE_LINKLINK', 'false').lower() == 'true'
enable_numba = True
