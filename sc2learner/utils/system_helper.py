import socket
import os
import uuid

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


def get_actor_id():
    return os.getenv('SLURM_JOB_ID', 'PID' + str(get_pid()) + 'UUID' + str(uuid.uuid1()))


def get_manager_node_ip(node_ip=None):
    if 'SLURM_JOB_ID' not in os.environ:
        import logging
        logging.warning('We are not running on slurm!')
        return '127.0.0.1'
    if node_ip is None:
        node_ip = get_ip(True)
    learner_manager_ip_prefix = '.'.join(node_ip.split('.')[0:3])
    return MANAGER_NODE_TABLE[learner_manager_ip_prefix]


if __name__ == "__main__":
    print('ip', get_ip())
    print('pid', get_pid())
    print('actor id', get_actor_id())
