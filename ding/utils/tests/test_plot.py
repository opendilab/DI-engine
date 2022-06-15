import os

import numpy as np
import pytest

from ding.utils.plot_helper import plot
from hbutils.testing import isolated_directory

@pytest.mark.unittest
def test_plot():
    rewards1 = np.array([0, 0.1, 0, 0.2, 0.4, 0.5, 0.6, 0.9, 0.9, 0.9])
    rewards2 = np.array([0, 0, 0.1, 0.4, 0.5, 0.5, 0.55, 0.8, 0.9, 1])
    rewards = np.concatenate((rewards1, rewards2))  # concatenation array
    episode1 = range(len(rewards1))
    episode2 = range(len(rewards2))
    episode = np.concatenate((episode1, episode2))
    data1 = {}
    data1['x'] = episode
    data1['y'] = rewards
    data1['label'] = 'line1'

    rewards3 = np.random.random(10)
    rewards4 = np.random.random(10)
    rewards = np.concatenate((rewards3, rewards4))  # concatenation array
    episode3 = range(len(rewards1))
    episode4 = range(len(rewards2))
    episode = np.concatenate((episode3, episode4))
    data2 = {}
    data2['x'] = episode
    data2['y'] = rewards
    data2['label'] = 'line2'

    data = [data1, data2]
    with isolated_directory():
        plot(data, 'step', 'reward_rate', 'test_pic', './pic.jpg')
        assert os.path.exists('./pic.jpg')
