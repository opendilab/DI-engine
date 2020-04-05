import os
import sys

from utils import read_file_ceph, save_file_ceph


def get_data():
    ceph_root = "s3://alphastar_fake_data/"
    traj_path = "model_iterations_2200.pth.tar_job_9af80bde-742c-11ea-a290-b50c26f99055_agent_0_step_16_0b42a67a-7436-11ea-a449-434cd50af7c8.traj"
    trajectory = read_file_ceph(ceph_root + traj_path, read_type='pickle')
    print("original size = {}".format(sys.getsizeof(trajectory)))
    return trajectory


if __name__ == '__main__':
    trajectory = get_data()


