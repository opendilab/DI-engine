import os
from easydict import EasyDict
from nervex.interaction.base import split_http_address

DEFAULT_NAMESPACE = 'default'
DEFAULT_POD_NAME = 'nervexjob-example-coordinator'


def get_operator_server_kwargs(cfg: EasyDict) -> dict:
    namespace = os.environ.get('KUBERNETES_POD_NAMESPACE', DEFAULT_NAMESPACE)
    name = os.environ.get('KUBERNETES_POD_NAME', DEFAULT_POD_NAME)
    host, port, _, _ = split_http_address(cfg.system_addr)
    return {
        'api_version': cfg.api_version,
        'namespace': namespace,
        'name': name,
        'host': host,
        'port': port,
    }
