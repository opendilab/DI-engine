import os
import sys
import subprocess
import requests
import torch

learner_port = '18193'
coordinator_ip = '10.5.36.31'
coordinator_port = '18194'
manager_ip = '10.5.36.31'
manager_port = '18195'
league_manager_port = '18196'

url_prefix_format = 'http://{}:{}/'

manager_uid = 'fake_manager_uid'
actor_uid = 'fake_actor_uid'
job_id = '8d2e8eda-83d9-11ea-8bb0-1be4f1872daf'
trajectory_path = 's3://alphastar_fake_data/model_main_player_zerg_0_ckpt.' \
    'pth_job_1a572940-838d-11ea-9feb-8f9bf499ec33_agent_0_step_2332_7df970f2-838d-11ea-86b6-355c4fbf42b9.traj'
learner_uid = '8d2a0fb8-83d9-11ea-bdab-1d045ae924a7'

metadata = {
    'job_id': job_id,
    'trajectory_path': trajectory_path,
    'learner_uid': learner_uid,
    'data': torch.tensor([[1, 2, 3], [4, 5, 6]]).tolist(),
    'step_data_compressor': 'lz4'
}


def get_url_prefix(ip, port):
    return url_prefix_format.format(ip, port)


def test_coordinator_register_manager():
    url_prefix = get_url_prefix(coordinator_ip, coordinator_port)
    d = {'manager_uid': manager_uid}
    response = requests.post(url_prefix + 'coordinator/register_manager', json=d).json()
    print(response)


def test_coordinator_ask_for_job():
    url_prefix = get_url_prefix(coordinator_ip, coordinator_port)
    d = {"manager_uid": manager_uid, "actor_uid": actor_uid}
    response = requests.post(url_prefix + 'coordinator/ask_for_job', json=d).json()
    print(response)


def test_coordinator_get_metadata():
    url_prefix = get_url_prefix(coordinator_ip, coordinator_port)
    d = {"manager_uid": manager_uid, "actor_uid": actor_uid, "job_id": job_id, "metadata": metadata}
    response = requests.post(url_prefix + 'coordinator/get_metadata', json=d).json()
    print(response)


def test_coordinator_finish_job():
    url_prefix = get_url_prefix(coordinator_ip, coordinator_port)
    d = {"manager_uid": manager_uid, "actor_uid": actor_uid, "job_id": job_id, "result": 'wins'}
    response = requests.post(url_prefix + 'coordinator/finish_job', json=d).json()
    print(response)


def test_coordinator_push_data_to_replay_buffer():
    url_prefix = get_url_prefix(coordinator_ip, coordinator_port)
    d = {'learner_uid': '3578336'}
    requests.post(url_prefix + 'debug/push_data_to_replay_buffer', json=d)


def test_read_file():
    from nervex.utils import read_file, get_step_data_decompressor
    decompressor = get_step_data_decompressor('lz4')
    x = read_file(trajectory_path)
    x = decompressor(x)
    print(x)


def read_resume():
    p = '../experiments/alphastar_rl_baseline/api-log/coordinator.resume.2020-05-04-23-18-31'
    data = torch.load(p)
    print(data)


if __name__ == '__main__':
    # test_coordinator_register_manager()
    # test_coordinator_ask_for_job()
    # test_coordinator_get_metadata()
    # test_coordinator_finish_job()
    test_coordinator_push_data_to_replay_buffer()
    # test_read_file()
    # read_resume()
