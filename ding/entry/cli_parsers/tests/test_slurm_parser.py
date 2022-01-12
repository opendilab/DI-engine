import pytest
import os
from ding.entry.cli_parsers import PLATFORM_PARSERS
slurm_parser = PLATFORM_PARSERS["slurm"]


@pytest.fixture
def set_slurm_env():
    os.environ["SLURM_NTASKS"] = '4'  # 参数 n，总进程/任务数
    os.environ["SLURM_NTASKS_PER_NODE"] = '2'  # 参数 ntasks-per-node，每个节点的进程数
    os.environ["SLURM_NODELIST"] = 'SH-IDC1-10-5-38-[190,215]'  # 所有节点
    os.environ["SLURM_SRUN_COMM_PORT"] = '42932'  # 哪些可用端口？
    os.environ["SLURM_TOPOLOGY_ADDR"] = 'SH-IDC1-10-5-38-215'  # 当前节点名
    os.environ["SLURM_NODEID"] = '1'  # 节点顺序，从 0 开始
    os.environ["SLURM_PROCID"] = '2'  # 进程顺序，从 0 开始，实际启动顺序可能与数字顺序不同
    os.environ["SLURM_LOCALID"] = '0'  # 本地顺序，从 0 开始，最大 ntasks-per-node - 1
    os.environ["SLURM_GTIDS"] = '2,3'  # 在当前进程上启动的 procid
    os.environ["SLURMD_NODENAME"] = 'SH-IDC1-10-5-38-215'  # 当前节点名

    yield

    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_NTASKS_PER_NODE"]
    del os.environ["SLURM_NODELIST"]
    del os.environ["SLURM_SRUN_COMM_PORT"]
    del os.environ["SLURM_TOPOLOGY_ADDR"]
    del os.environ["SLURM_NODEID"]
    del os.environ["SLURM_PROCID"]
    del os.environ["SLURM_LOCALID"]
    del os.environ["SLURM_GTIDS"]
    del os.environ["SLURMD_NODENAME"]


@pytest.mark.unittest
@pytest.mark.usefixtures('set_slurm_env')
def test_slurm_parser():
    platform_spec = {
        "type": "slurm",
        "tasks": [
            {
                "labels": "league,collect",
                "node_ids": 10,
            }, {
                "labels": "evaluate",
                "node_ids": 20,
                "attach_to": "$node.10"
            }, {
                "labels": "learn",
                "node_ids": 31,
                "attach_to": "$node.10,$node.20"
            }, {
                "labels": "learn",
                "node_ids": 32,
                "attach_to": "$node.10,$node.20"
            }
        ]
    }
    all_args = slurm_parser(platform_spec)
    assert all_args["labels"] == "learn"
    assert all_args["address"] == "SH-IDC1-10-5-38-215"
    assert all_args["ports"] == 15151  # Start from 15151
    assert all_args["node_ids"] == 31
    assert all_args["attach_to"] == "tcp://SH-IDC1-10-5-38-190:15151,tcp://SH-IDC1-10-5-38-190:15152"
