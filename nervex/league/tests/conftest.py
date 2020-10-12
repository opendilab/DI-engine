import numpy as np
import pytest


@pytest.fixture(scope='session')
def random_task_result():

    def fn():
        p = np.random.uniform()
        if p < 1. / 3:
            return "wins"
        elif p < 2. / 3:
            return "draws"
        else:
            return "losses"

    return fn


@pytest.fixture(scope='session')
def get_task_result_categories():
    return ["wins", 'draws', 'losses']
