import os
import sys
import subprocess

# from utils import read_file_ceph, save_file_ceph

# def get_data():
#     ceph_root = "s3://alphastar_fake_data/"
#     traj_path = "model_iterations_2200.pth.tar_job_9af80bde-742c-11ea-a290-b50c26f99055_agent_0_step_16_0b42a67a-7436-11ea-a449-434cd50af7c8.traj"
#     trajectory = read_file_ceph(ceph_root + traj_path, read_type='pickle')
#     print("original size = {}".format(sys.getsizeof(trajectory)))
#     return trajectory


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


def generate():
    p = 'nohup srun -p {} -w {} python -u replay_fix.py {} > {} 2>&1 &'
    list_p = '/mnt/lustre/zhangming/data/listtempfix_2/todo.list.'
    log_p = '/mnt/lustre/zhangming/data/logfix_2/log.'
    d = get_cls_info()
    partitions = ['VI_SP_Y_V100_A', 'VI_SP_VA_V100', 'VI_SP_VA_1080TI', 'VI_ID_1080TI']

    count = 0
    for partition in partitions:
        workstations = d[partition]
        for workstation_state in workstations:
            workstation = workstation_state[0]
            print(p.format(partition, workstation, list_p + "{:02d}".format(count), log_p + "{:02d}".format(count)))
            count += 1
            if count >= 50:
                exit()


if __name__ == '__main__':
    # trajectory = get_data()
    generate()
