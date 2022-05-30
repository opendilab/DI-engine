import os
import subprocess
from typing import Optional, Dict, Tuple

MANAGER_NODE_TABLE = {
    '10.198.8': '10.198.8.31',
    '10.198.6': '10.198.6.31',
    '10.5.38': '10.5.38.31',
    '10.5.39': '10.5.38.31',
    '10.5.36': '10.5.36.31',
    '10.5.37': '10.5.36.31',
    '10.10.30': '10.10.30.91',
}


def get_ip() -> str:
    assert os.environ.get('SLURMD_NODENAME'), 'not found SLURMD_NODENAME env variable'
    # expecting nodename to be like: 'SH-IDC1-10-5-36-64'
    nodename = os.environ.get('SLURMD_NODENAME', '')
    myaddr = '.'.join(nodename.split('-')[-4:])
    return myaddr


def get_manager_node_ip(node_ip: Optional[str] = None) -> str:
    r"""
    Overview:
        Look up the manager node of the slurm cluster and return the node ip
    """
    if 'SLURM_JOB_ID' not in os.environ:
        from ditk import logging
        logging.error(
            'We are not running on slurm!, \'auto\' for manager_ip or '
            'coordinator_ip is only intended for running on multiple slurm clusters'
        )
        return '127.0.0.1'
    node_ip = node_ip or get_ip()
    learner_manager_ip_prefix = '.'.join(node_ip.split('.')[0:3])

    if learner_manager_ip_prefix in MANAGER_NODE_TABLE:
        return MANAGER_NODE_TABLE[learner_manager_ip_prefix]
    else:
        raise KeyError("Cluster not found, please add it to the MANAGER_NODE_TABLE in {}".format(__file__))


# get all info of cluster
def get_cls_info() -> Dict[str, list]:
    ret_dict = {}
    info = subprocess.getoutput('sinfo -Nh').split('\n')
    for line in info:
        line = line.strip().split()
        if len(line) != 4:
            continue
        node, _, partition, state = line
        if partition not in ret_dict:
            ret_dict[partition] = []
        assert node not in ret_dict[partition]
        if state in ['idle', 'mix']:
            ret_dict[partition].append(node)

    return ret_dict


def node_to_partition(target_node: str) -> Tuple[str, str]:
    info = subprocess.getoutput('sinfo -Nh').split('\n')
    for line in info:
        line = line.strip().split()
        if len(line) != 4:
            continue
        node, _, partition, state = line
        if node == target_node:
            return partition
    raise RuntimeError("not found target_node: {}".format(target_node))


def node_to_host(node: str) -> str:
    return '.'.join(node.split('-')[-4:])


def find_free_port_slurm(node: str) -> int:
    partition = node_to_partition(node)
    if partition == 'spring_scheduler':
        comment = '--comment=spring-submit'
    else:
        comment = ''
    output = subprocess.getoutput(
        "srun -p {} -w {} {} python -c \"from ding.utils import find_free_port; print('port' + str(find_free_port(0)))\""  # noqa
        .format(partition, node, comment)
    )
    port = output.split('port')[-1]
    return int(port)
