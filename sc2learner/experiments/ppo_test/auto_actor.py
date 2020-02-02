# -*- coding: UTF-8 -*-
import os
import sys
import time
import socket
import subprocess
import yaml
import random
from easydict import EasyDict
from collections import defaultdict

# as squeue only show 8 char of name
me = subprocess.getoutput('whoami')[:8]
mefull = subprocess.getoutput('whoami')

# parameters determining the policy of starting actors
CPU_IDLE = 20
OUR_IDLE = 20
OTHER_IDLE = 15
CPU_MIX = 20
OUR_MIX = 10
OTHER_MIX = 5

partitions_dict = {
    '10.10.30.91': {
        'cpu'    : [],
        'ours'   : ['VI_SP_Y_V100'],
        'others' : ['VI_IPS_V', 'VI_ID_1080', 'VI_ID_V100', 'VI_IPS_1080', 'VI_SP_X_TITANX']
    },
    '10.5.36.31': {
        'cpu'    : [],
        'ours'   : ['VI_SP_Y_V100_A'],
        'others' : ['VI_SP_Y_V100_B', 'VI_Face_1080TI', 'VI_Face_V100', 'VI_ID_1080TI', 'VI_SP_VA_1080TI', 'VI_SP_VA_V100']
    },
    '10.5.38.31': {
        'cpu'    : [],
        'ours'   : ['VI_SP_Y_1080TI'],
        'others' : ['VI_ID_1080TI', 'VI_IPS_1080TI', 'VI_SP_X_1080TI', 'VI_SP_Z_1080TI', 'VI_UC_1080TI']
    },
    '10.198.6.31': {
        'cpu'    : ['cpu'],
        'ours'   : ['x_cerebra'],
        'others' : ['sensetime', 'scg_industry_v100', 'ips_share', 'ucg', 'vi_face_v100_32g', 'vi_id_v100']
    },
    '10.198.8.31': {
        'cpu'    : [],
        'ours'   : ['x_cerebra'],
        'others' : []
    }
}

node_prefix_dict = {
    '10.10.30.91': 'BJ-IDC1-',
    '10.5.36.31': 'SH-IDC1-',
    '10.5.38.31': 'SH-IDC1-',
    '10.198.6.31': 'SH-IDC1-',
    '10.198.8.31': 'SH-IDC1-',
}

def node_liqiao(node_address):
    actor_num = 0
    is_learner = False
    info = subprocess.getoutput('squeue -h -w {}'.format(node_address)).split('\n')
    if len(info) > 0:
        for line in info:
            # print(line)
            job_id, partition, job_name, user, state, run_time, nodes, node_list = line.strip().split()
            if job_name == 'actor' and user == me:
                actor_num += 1
            if job_name == 'learner' and user == me:
                is_learner = True
    return actor_num, is_learner


def scancel(job_id):
    info = subprocess.run(['scancel', str(job_id)])


def launch(partition, node_address, num=1):
    subprocess.run(['sh', 'actor.sh', partition, node_address, str(num)])


def pd_partition(partition, p_class, limit, learner_node_address, actor_manager_node_address):
    actor_num = 0

    if p_class == 'cpu':
        mix_actor_num = CPU_MIX
        idle_actor_num = CPU_IDLE
    elif p_class == 'our':
        mix_actor_num = OUR_MIX
        idle_actor_num = OUR_IDLE
    elif p_class == 'other':
        mix_actor_num = OTHER_MIX
        idle_actor_num = OTHER_IDLE
    else:
        raise Exception

    info = subprocess.getoutput('sinfo -Nhp {} | sort -k4'.format(partition)).split('\n')
    idle_list = []
    mix_list = []
    for line in info:
        line = line.strip().split()
        if len(line) != 4:
            continue
        node_address, num, _, state = line
        #skipping learner node or actor_manager_node
        if learner_node_address != node_address and actor_manager_node_address != node_address:
            if state == 'mix':
                mix_list.append(node_address)
            elif state == 'idle':
                idle_list.append(node_address)

    # idle
    for node_address in idle_list:
        launch(partition, node_address, num=idle_actor_num)
        actor_num += idle_actor_num
        if limit < actor_num:
            break

    info = subprocess.getoutput('squeue -hp {}'.format(partition)).split('\n')
    for line in info:
        line = line.strip().split()
        if len(line) != 8:
            continue
        job_id, _, job_name, user, state, run_time, nodes, node_list = line
        if user == me and state == 'PD' and job_name == 'actor':
            scancel(job_id)
            actor_num -= 1
    if limit < actor_num:
        return actor_num

    # mix
    for node_address in mix_list:
        launch(partition, node_address, num=mix_actor_num)
        actor_num += mix_actor_num
        if limit < actor_num:
            break

    info = subprocess.getoutput('squeue -hp {}'.format(partition)).split('\n')
    for line in info:
        line = line.strip().split()
        if len(line) != 8:
            continue
        job_id, _, job_name, user, state, run_time, nodes, node_list = line
        if user == me and state == 'PD' and job_name == 'actor':
            scancel(job_id)
            actor_num -= 1
    if limit < actor_num:
        return actor_num

    return actor_num


def run_manager(ip, cfg):
    if cfg.communication.ip.actor_manager != 'auto':
        print('run actor manager on specified node (not on manager node)')
        node_name = node_prefix_dict[ip] + '-'.join(cfg.communication.ip.actor_manager.split('.'))
        manager_partition = None
        info = subprocess.getoutput('sinfo -Nh').split('\n')
        for line in info:
            line = line.strip().split()
            if len(line) != 4:
                continue
            node_address, num, partition_name, state = line
            if node_address == node_name:
                manager_partition = partition_name
        assert manager_partition != None, 'cannot find the partition of the actor_manager node'
        subprocess.run(['sh', 'actor_manager_queue.sh', manager_partition, node_name, mefull, '&'])
        time.sleep(30)
    else:
        print('run manager on manager node')
        subprocess.run(['sh', 'actor_manager.sh', '&'])
        time.sleep(30)


def get_learner():
    info = subprocess.getoutput('squeue -h').split('\n')
    for line in info:
        line = line.strip().split()
        if len(line) != 8:
            continue
        job_id, partition, job_name, user, state, run_time, nodes, node_list = line
        if job_name == 'learner' and user == me:
            return node_list
    return None

def get_actor_manager_local():
    info = subprocess.getoutput('squeue -h').split('\n')
    for line in info:
        line = line.strip().split()
        if len(line) != 8:
            continue
        job_id, partition, job_name, user, state, run_time, nodes, node_list = line
        if job_name == 'actor_ma' and user == me:
            return node_list
    return None

def main(actor_limit, manager_flag=0):
    # get lustre ip
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    with open('config.yaml') as f:
        cfg =yaml.load(f)
    cfg = EasyDict(cfg)

    # run manager
    if manager_flag:
        run_manager(ip, cfg)
    else:
        print('no run manager')

    learner_node_address = get_learner()
    actor_manager_node_address = get_actor_manager_local()

    assert ip in partitions_dict.keys(), 'ip must be one of [{}]'.format(', '.join(partitions_dict.keys()))
    cpu_partitions = partitions_dict[ip]['cpu']
    our_partitions = partitions_dict[ip]['ours']
    other_partitions = partitions_dict[ip]['others']
    random.shuffle(other_partitions)

    actor_num_touse = actor_limit
    actor_num_all = 0
    actor_num_cpu = 0
    actor_num_our = 0
    actor_num_other = 0

    for partition in cpu_partitions:
        if actor_num_touse <= 0:
            break
        actor_num = pd_partition(partition, 'cpu', actor_num_touse, learner_node_address, actor_manager_node_address)
        actor_num_cpu += actor_num
        actor_num_touse -= actor_num
    actor_num_all += actor_num_cpu

    for partition in our_partitions:
        if actor_num_touse <= 0:
            break
        actor_num = pd_partition(partition, 'our', actor_num_touse, learner_node_address, actor_manager_node_address)
        actor_num_our += actor_num
        actor_num_touse -= actor_num
    actor_num_all += actor_num_our

    for partition in other_partitions:
        if actor_num_touse <= 0:
            break
        actor_num += pd_partition(partition, 'other', actor_num_touse, learner_node_address, actor_manager_node_address)
        actor_num_other += actor_num
        actor_num_touse -= actor_num
    actor_num_all += actor_num_other

    if actor_num_all < actor_limit:
        print('Warning: cannot start required number of actors!')
    print(' cpu: {} \n our: {} \n other: {} \n total: {} \n limit: {} \n'.format(actor_num_cpu, actor_num_our, actor_num_other, actor_num_all, actor_limit))

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))

