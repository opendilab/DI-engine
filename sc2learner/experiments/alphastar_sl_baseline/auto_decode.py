# -*- coding: UTF-8 -*-
import os
import sys
import time
import subprocess

# as squeue only show 8 char of name
me = subprocess.getoutput('whoami')[:8]


def launch(partition, workstation, replay_name, output_dir, log_dir):
    log_path = os.path.join(log_dir, 'log.' + replay_name.split('/')[-1])
    subprocess.run(['bash', 'decode.sh', partition, workstation, replay_name, output_dir, log_path])


# get all info of cluster
def get_cls_info():
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

# get my task number by node


def get_workstation_info(partition, workstation):
    info = subprocess.getoutput('squeue -h -p {} -w {}'.format(partition, workstation)).split('\n')
    count = 0
    for line in info:
        line = line.strip().split()
        if len(line) != 8:
            continue
        job_id, partition, job_name, user, state, run_time, nodes, node_list = line
        if user == me:
            count += 1
    return count


def check_if_exist(output_dir, replay_name_prefix):
    for f in os.listdir(output_dir):
        if replay_name_prefix in f:
            return True
    return False


def main():
    workers_num_std = 2
    log_dir = '/mnt/lustre/zhangming/data/logs_valid'
    output_dir = '/mnt/lustre/zhangming/data/Replays_decode_valid'
    partitions = ['VI_SP_Y_V100_B', 'VI_SP_Y_V100_A', 'VI_SP_VA_V100',
                  'VI_SP_VA_1080TI', 'VI_Face_1080TI', 'VI_ID_1080TI']
    #partitions = ['VI_SP_Y_V100_A']
    info = get_cls_info()
    replays = open('/mnt/lustre/zhangming/data/listtemp/replays.valid.list.06', 'r').readlines()
    replays = [x.strip() for x in replays]
    replays_index = 0
    while True:
        for partition in partitions:
            workstations = []
            if partition != 'VI_SP_Y_V100_A':
                workers_num_std_spe = 1
                workstations = info[partition][:2]
            else:
                workers_num_std_spe = workers_num_std
                workstations = info[partition]
            for workstation in workstations:
                workers_num = get_workstation_info(partition, workstation)
                if workers_num_std_spe - workers_num > 0:
                    for i in range(workers_num_std_spe - workers_num):
                        replay_name = replays[replays_index]
                        replay_name_prefix = replay_name.split('/')[-1].split(".")[0]
                        while check_if_exist(output_dir, replay_name_prefix):
                            print('skip {} {}'.format(replays_index, replay_name))
                            replays_index += 1
                            if replays_index >= len(replays):
                                return
                            replay_name = replays[replays_index]
                            replay_name_prefix = replay_name.split(".")[0]

                        launch(partition, workstation, replay_name, output_dir, log_dir)
                        print('{} {}'.format(replays_index, replay_name))
                        replays_index += 1
                        if replays_index >= len(replays):
                            return
        time.sleep(600)


if __name__ == '__main__':
    main()
