import os
from easydict import EasyDict
from nervex.interaction.base import split_http_address

DEFAULT_NAMESPACE = 'default'
DEFAULT_POD_NAME = 'nervexjob-example-coordinator'


def get_operator_server_kwargs(cfg: EasyDict) -> dict:
    r'''
    Overview:
        Get kwarg dict from config file
    Arguments:
        - cfg (:obj:`EasyDict`) System config
    Returns:
        - result (:obj:`dict`) Containing ``api_version``,  ``namespace``, ``name``, ``port``, ``host``.
    '''
    namespace = os.environ.get('KUBERNETES_POD_NAMESPACE', DEFAULT_NAMESPACE)
    name = os.environ.get('KUBERNETES_POD_NAME', DEFAULT_POD_NAME)
    url = cfg.get('system_addr', None) or os.environ.get('KUBERNETES_NERVEX_SERVER_URL', None)
    assert url, 'please set environment variable KUBERNETES_NERVEX_SERVER_URL in Kubenetes platform.'
    host, port, _, _ = split_http_address(url)
    return {
        'api_version': cfg.api_version,
        'namespace': namespace,
        'name': name,
        'host': host,
        'port': port,
    }
