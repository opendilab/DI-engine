from .slurm_parser import slurm_parser
from .k8s_parser import k8s_parser
PLATFORM_PARSERS = {"slurm": slurm_parser, "k8s": k8s_parser}
