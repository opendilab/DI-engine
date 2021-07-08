import os

from easydict import EasyDict

from ding.interaction.base import split_http_address

DEFAULT_NAMESPACE = 'default'
DEFAULT_POD_NAME = 'dijob-example-coordinator'
DEFAULT_API_VERSION = '/v1alpha1'

DEFAULT_K8S_COLLECTOR_PORT = 22270
DEFAULT_K8S_LEARNER_PORT = 22271
DEFAULT_K8S_AGGREGATOR_SLAVE_PORT = 22272
DEFAULT_K8S_COORDINATOR_PORT = 22273
DEFAULT_K8S_AGGREGATOR_MASTER_PORT = 22273


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
    url = cfg.get('system_addr', None) or os.environ.get('KUBERNETES_SERVER_URL', None)
    assert url, 'please set environment variable KUBERNETES_SERVER_URL in Kubenetes platform.'
    api_version = cfg.get('api_version', None) \
        or os.environ.get('KUBERNETES_SERVER_API_VERSION', DEFAULT_API_VERSION)
    try:
        host, port = url.split(":")[0], int(url.split(":")[1])
    except Exception as e:
        host, port, _, _ = split_http_address(url)

    return {
        'api_version': api_version,
        'namespace': namespace,
        'name': name,
        'host': host,
        'port': port,
    }


def exist_operator_server() -> bool:
    return 'KUBERNETES_SERVER_URL' in os.environ
