from time import sleep
import pytest
import pathlib
import os
from ding.entry.cli_ditask import _cli_ditask


def cli_ditask_main():
    sleep(0.1)


@pytest.mark.unittest
def test_cli_ditask():
    kwargs = {
        "package": os.path.dirname(pathlib.Path(__file__)),
        "main": "test_cli_ditask.cli_ditask_main",
        "parallel_workers": 1,
        "topology": "mesh",
        "platform": "k8s",
        "protocol": "tcp",
        "ports": 50501,
        "attach_to": "",
        "address": "127.0.0.1",
        "labels": "",
        "node_ids": 0,
        "mq_type": "nng",
        "redis_host": "",
        "redis_port": ""
    }
    os.environ["DI_NODES"] = '127.0.0.1'
    os.environ["DI_RANK"] = '0'
    try:
        _cli_ditask(**kwargs)
    finally:
        del os.environ["DI_NODES"]
        del os.environ["DI_RANK"]
