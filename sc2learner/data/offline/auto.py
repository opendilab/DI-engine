# -*- coding: UTF-8 -*-
import os
import sys
import time
import subprocess

# as squeue only show 8 char of name
me = subprocess.getoutput('whoami')[:8]
finish_list = []

def launch(partition, workstation, replay_name, output_dir, parallel, log_dir):
    log_path = os.path.join(log_dir, 'log.' + workstation)
    subprocess.run(['bash', 'replay_decode_game_loop.sh', partition, workstation, replay_name, output_dir, parallel, log_path]) 


def launch_check(partition, workstation, list_path, log_dir):
    log_path = os.path.join(log_dir, 'log.' + workstation)
    subprocess.run(['bash', 'check.sh', partition, workstation, list_path, log_path]) 


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
            ret_dict[partition].append([node, state])

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
    if finish_list == []:
        for f in os.listdir(output_dir):
            finish_list.append(f)
    for f in finish_list:
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


def main_4_8():
    workers_num_std = 2
    workers_num_my = 6
    delete_workstations = ['SH-IDC1-10-5-36-158']
    list_dir = '/mnt/lustre/zhangming/data/listtemp48'
    log_dir = '/mnt/lustre/zhangming/data/logs_valid_48'
    output_dir = '/mnt/lustre/zhangming/data/Replays_decode_valid_48'
    if not os.path.isdir(list_dir):
        os.mkdir(list_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    partitions = ['VI_SP_Y_V100_B', 'VI_SP_Y_V100_A', 'VI_SP_VA_V100',
                  'VI_SP_VA_1080TI', 'VI_ID_1080TI', 'VI_Face_1080TI']
    # partitions = ['VI_SP_Y_V100_A']
    info = get_cls_info()
    total_list_num = 0
    for k, v in info.items():
        if k in partitions:
            total_list_num += len(v)
            for delete_workstation in delete_workstations:
                if delete_workstation in v:
                    total_list_num -= 1
    replays = open('/mnt/lustre/zhangming/data/listtemp48/replays.raw.list.0.1353962', 'r').readlines()
    print('total: {}'.format(len(replays)))
    replays = [x.strip() for x in replays if not check_if_exist(output_dir, x.strip().split('/')[-1].split('.')[0])]
    print('todo: {}'.format(len(replays)))
    len_per_workstation = len(replays) // total_list_num + 1
    print(total_list_num, len_per_workstation)
    replays_index = 0

    for partition in partitions:
        replay_list_path = os.path.join(list_dir, partition)
        if not os.path.isdir(replay_list_path):
            os.mkdir(replay_list_path)
        workstations = []
        if partition != 'VI_SP_Y_V100_A':
            workers_num_std_spe = workers_num_std
            workstations = info[partition]
        else:
            workers_num_std_spe = workers_num_my
            workstations = info[partition]
        for workstation_state in workstations:
            workstation = workstation_state[0]
            state = workstation_state[0]
            if workstation not in delete_workstations:
                replay_list_path_now = os.path.join(replay_list_path, workstation)
                with open(replay_list_path_now, 'w') as f:
                    for replay in replays[replays_index*len_per_workstation:(replays_index+1)*len_per_workstation]:
                        f.write(replay + '\n')
                if state == 'idle':
                    workers_num_std_spe = workers_num_my
                launch(partition, workstation, replay_list_path_now, output_dir, str(workers_num_std_spe), log_dir)
                print('{} {}'.format(replays_index, replay_list_path_now))
                replays_index += 1


def main_check():
    workers_num_std = 1
    workers_num_my = 1
    # delete_workstations = ['SH-IDC1-10-5-36-152', 'SH-IDC1-10-5-36-158', 'SH-IDC1-10-5-37-18',
    #                       'SH-IDC1-10-5-37-28', 'SH-IDC1-10-5-37-37', 'SH-IDC1-10-5-36-149']
    delete_workstations = []
    list_dir = '/mnt/lustre/zhangming/data/listtemp410check'
    log_dir = '/mnt/lustre/zhangming/data/logs_valid_410_check'
    if not os.path.isdir(list_dir):
        os.mkdir(list_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    partitions = ['VI_SP_Y_V100_B', 'VI_SP_Y_V100_A', 'VI_SP_VA_V100',
                  'VI_SP_VA_1080TI', 'VI_ID_1080TI']
    # partitions = ['VI_SP_Y_V100_A']
    info = get_cls_info()
    total_list_num = 0
    for k, v in info.items():
        if k in partitions:
            total_list_num += len(v)
            for delete_workstation in delete_workstations:
                if delete_workstation in v:
                    total_list_num -= 1
    replays = open('/mnt/lustre/zhangming/data/replay_decode_410_clean_todo.list', 'r').readlines()
    print('total: {}'.format(len(replays)))
    len_per_workstation = len(replays) // total_list_num + 1
    print(total_list_num, len_per_workstation)
    replays_index = 0

    for partition in partitions:
        replay_list_path = os.path.join(list_dir, partition)
        if not os.path.isdir(replay_list_path):
            os.mkdir(replay_list_path)
        workstations = []
        if partition != 'VI_SP_Y_V100_A':
            workers_num_std_spe = workers_num_std
            workstations = info[partition]
        else:
            workers_num_std_spe = workers_num_my
            workstations = info[partition]
        for workstation_state in workstations:
            workstation = workstation_state[0]
            state = workstation_state[0]
            if workstation not in delete_workstations:
                replay_list_path_now = os.path.join(replay_list_path, workstation)
                with open(replay_list_path_now, 'w') as f:
                    for replay in replays[replays_index*len_per_workstation:(replays_index+1)*len_per_workstation]:
                        f.write(replay.strip() + '\n')
                launch_check(partition, workstation, replay_list_path_now, log_dir)
                print('{} {}'.format(replays_index, replay_list_path_now))
                replays_index += 1

if __name__ == '__main__':
    func_name = sys.argv[1]
    if func_name == 'main_4_8':
        main_4_8()
    elif func_name == 'main_check':
        main_check()
