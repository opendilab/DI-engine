import pytest
import os
from ding.entry.cli_parsers.k8s_parser import k8s_parser


@pytest.fixture
def set_k8s_env():
    os.environ["DI_NODES"] = 'SH-0,SH-1,SH-2,SH-3,SH-4,SH-5'  # All the nodes
    os.environ["DI_RANK"] = '3'  # Proc order, start from 0, can not be modified by config

    yield

    del os.environ["DI_NODES"]
    del os.environ["DI_RANK"]


@pytest.mark.unittest
@pytest.mark.usefixtures('set_k8s_env')
def test_k8s_parser():
    # With platform_spec
    platform_spec = {
        "tasks": [
            {
                "labels": "league,collect",
                "node_ids": 10
            }, {
                "labels": "league,collect",
                "node_ids": 11
            }, {
                "labels": "evaluate",
                "node_ids": 20,
                "attach_to": "$node.10,$node.11"
            }, {
                "labels": "learn",
                "node_ids": 31,
                "ports": 50000,
                "attach_to": "$node.10,$node.11,$node.20"
            }, {
                "labels": "learn",
                "node_ids": 32,
                "attach_to": "$node.10,$node.11,$node.20"
            }, {
                "labels": "learn",
                "node_ids": 33,
                "attach_to": "$node.10,$node.11,$node.20"
            }
        ]
    }
    all_args = k8s_parser(platform_spec, mq_type="nng")
    assert all_args["labels"] == "learn"
    assert all_args["address"] == "SH-3"
    assert all_args["ports"] == 50000
    assert all_args["node_ids"] == 31
    assert all_args["parallel_workers"] == 1
    assert all_args[
        "attach_to"
    ] == "tcp://SH-0:50515," +\
        "tcp://SH-1:50515," +\
        "tcp://SH-2:50515"

    # Without platform_spec, parse by global config
    all_args = k8s_parser(None, topology="mesh", mq_type="nng")
    assert all_args["address"] == "SH-3"
    assert all_args["node_ids"] == 3
    assert all_args["parallel_workers"] == 1
    assert all_args[
        "attach_to"
    ] == "tcp://SH-0:50515," +\
        "tcp://SH-1:50515," +\
        "tcp://SH-2:50515"

    # With multiple parallel workers
    all_args = k8s_parser(None, topology="mesh", parallel_workers=2)
    assert all_args["address"] == "SH-3"
    assert all_args["node_ids"] == 6
    assert all_args["parallel_workers"] == 2
    assert all_args[
        "attach_to"
    ] == "tcp://SH-0:50515,tcp://SH-0:50516," +\
        "tcp://SH-1:50515,tcp://SH-1:50516," +\
        "tcp://SH-2:50515,tcp://SH-2:50516"
