"""
Copyright 2020 Sensetime X-lab. All Rights Reserved
"""
import os
import socket
import time
import uuid
from typing import Optional
from contextlib import closing


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
    r"""
    Overview:
        get the ip of slurm/socket
    Returns:
        - ip(:obj:`str`): the corresponding ip
    """
    if os.environ.get('SLURMD_NODENAME'):
        # expecting nodename to be like: 'SH-IDC1-10-5-36-64'
        nodename = os.environ.get('SLURMD_NODENAME', '')
        myaddr = '.'.join(nodename.split('-')[-4:])
    else:
        # beware: return 127.0.0.1 on some slurm nodes
        myname = socket.getfqdn(socket.gethostname())
        myaddr = socket.gethostbyname(myname)

    return myaddr


def get_pid() -> int:
    r"""
    Overview:
        os.getpid
    """
    return os.getpid()


def get_task_uid() -> str:
    r"""
    Overview:
        get the slurm job_id, pid and uid
    """
    return os.getenv('SLURM_JOB_ID', 'PID{pid}UUID{uuid}'.format(
        pid=str(get_pid()),
        uuid=str(uuid.uuid1()),
    )) + '_' + str(time.time())


def get_manager_node_ip(node_ip: Optional[str] = None) -> str:
    r"""
    Overview:
        Look up the manager node of the slurm cluster and return the node ip
    """
    if 'SLURM_JOB_ID' not in os.environ:
        import logging
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


def find_free_port():
    r"""
    Overview:
        Look up the free port list and return one
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

