import pytest
import logging
import time
from ding.utils import log_every_n, log_every_sec


@pytest.mark.unittest
def test_sparse_logging():
    logging.getLogger().setLevel(logging.INFO)
    for i in range(30):
        log_every_n(logging.INFO, 5, "abc_{}".format(i))

    for i in range(30):
        time.sleep(0.1)
        log_every_sec(logging.INFO, 1, "abc_{}".format(i))
