import os
import json
from typing import Tuple
from easydict import EasyDict
from ding.interaction.base import split_http_address
from .default_helper import one_time_warning

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


def pod_exec_command(kubeconfig: str, name: str, namespace: str, cmd: str) -> Tuple[int, str]:
    try:
        from kubernetes import config
        from kubernetes.client import CoreV1Api
        from kubernetes.client.rest import ApiException
        from kubernetes.stream import stream
    except ModuleNotFoundError as e:
        one_time_warning("You have not installed kubernetes package! Please try 'pip install DI-engine[k8s]'.")
        exit(-1)

    config.load_kube_config(config_file=kubeconfig)
    core_v1 = CoreV1Api()
    resp = None
    try:
        resp = core_v1.read_namespaced_pod(name=name, namespace=namespace)
    except ApiException as e:
        if e.status != 404:
            return -1, "Unknown error: %s" % e
    if not resp:
        return -1, f"Pod {name} does not exist."
    if resp.status.phase != 'Running':
        return -1, f"Pod {name} is not in Running."
    exec_command = ['/bin/sh', '-c', cmd]
    resp = stream(
        core_v1.connect_get_namespaced_pod_exec,
        name,
        namespace,
        command=exec_command,
        stderr=False,
        stdin=False,
        stdout=True,
        tty=False
    )
    resp = resp.replace("\'", "\"") \
        .replace('None', 'null') \
        .replace(': False', ': 0') \
        .replace(': True', ': 1') \
        .replace('"^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$"', '\\"^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$\\"')
    resp = json.loads(resp)
    return resp['code'], resp['message']
