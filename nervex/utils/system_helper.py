import socket
import os
import uuid
import time

MANAGER_NODE_TABLE = {
    '10.198.8': '10.198.8.31',
    '10.198.6': '10.198.6.31',
    '10.5.38': '10.5.38.31',
    '10.5.39': '10.5.38.31',
    '10.5.36': '10.5.36.31',
    '10.5.37': '10.5.36.31',
    '10.10.30': '10.10.30.91',
}


def get_ip(on_slurm=True):
    if on_slurm:
        nodename = os.environ.get('SLURMD_NODENAME', '')
        # expecting nodename to be like: 'SH-IDC1-10-5-36-64'
        myaddr = '.'.join(nodename.split('-')[-4:])
    else:
        # beware: return 127.0.0.1 on some slurm nodes
        myname = socket.getfqdn(socket.gethostname())
        myaddr = socket.gethostbyname(myname)
    return myaddr


def get_pid():
    return os.getpid()


def get_task_uid():
    t = time.time()
    return os.getenv('SLURM_JOB_ID', 'PID' + str(get_pid()) + 'UUID' + str(uuid.uuid1())) + '_' + str(t)


def get_manager_node_ip(node_ip=None):
    '''Look up the manager node of the slurm cluster'''
    if 'SLURM_JOB_ID' not in os.environ:
        import logging
        logging.error(
            'We are not running on slurm!, \'auto\' for manager_ip or '
            'coordinator_ip is only intended for running on multiple slurm clusters'
        )
        return '127.0.0.1'
    if node_ip is None:
        node_ip = get_ip(True)
    learner_manager_ip_prefix = '.'.join(node_ip.split('.')[0:3])
    assert learner_manager_ip_prefix in MANAGER_NODE_TABLE,\
        'I don\'t know where is the manager node of this cluster'\
        'Please add it to the MANAGER_NODE_TABLE in {}'.format(__file__)
    return MANAGER_NODE_TABLE[learner_manager_ip_prefix]
