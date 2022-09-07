import os
import pytest
from copy import deepcopy
import numpy as np
import pandas as pd
from ding.entry.serial_entry import serial_pipeline
from dizoo.gym_anytrading.config import stocks_dqn_config, stocks_dqn_create_config


@pytest.mark.platformtest
@pytest.mark.unittest
def test_stocks_dqn():
    config = [deepcopy(stocks_dqn_config), deepcopy(stocks_dqn_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'stocks_dqn_unittest'
    config[0].env.stocks_data_filename = 'STOCKS_FAKE'

    # ======== generate fake data =========
    Date = pd.bdate_range(start='2010-02-20', end='2022-02-20')
    data = {'Date': [], 'Open': [], 'High': [], 'Low': [], 'Close': [], 'Adj Close': [], 'Volume': []}
    for i in range(len(Date)):
        data['Date'].append(Date[i])
        data['Low'].append(np.random.uniform(200, 500))
        data['High'].append(np.random.uniform(data['Low'][-1], data['Low'][-1] + 10))
        data['Open'].append(np.random.uniform(data['Low'][-1], data['High'][-1]))
        data['Close'].append(np.random.uniform(data['Low'][-1], data['High'][-1]))
        data['Adj Close'].append(data['Close'][-1])
        data['Volume'].append(np.random.randint(1000000, 7000000))
    # =====================================

    fake_data = pd.DataFrame(data)
    data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path += '/dizoo/gym_anytrading/envs/data/STOCKS_FAKE.csv'
    fake_data.to_csv(data_path, sep=',', index=None)
    try:
        serial_pipeline(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, "pipeline fail"
    finally:
        os.remove(data_path)
        os.popen('rm -rf {}'.format(os.path.abspath('./stocks_dqn_unittest')))
