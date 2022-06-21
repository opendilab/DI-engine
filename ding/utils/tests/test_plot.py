import os

import numpy as np
import pytest

from ding.utils.plot_helper import plot
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
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
        plt.figure(1)
        plot(data, 'step', 'reward_rate', 'test_pic', './pic.jpg')
        assert os.path.exists('./pic.jpg')
        plt.figure(2)
        sns.set(style="darkgrid", font_scale=1.5)
        sns.lineplot(x=np.concatenate((episode1, episode2)), y=np.concatenate((rewards1, rewards2)), label='line1')
        sns.lineplot(x=np.concatenate((episode3, episode4)), y=np.concatenate((rewards3, rewards4)), label='line2')
        plt.xlabel('step')
        plt.ylabel('reward_rate')
        plt.title('test_pic')
        plt.savefig('./pic_compare.jpg')
        I1 = Image.open('./pic.jpg')
        I2 = Image.open('./pic_compare.jpg')
        I1_array = np.array(I1)
        I2_array = np.array(I2)
        assert (I1_array - I2_array).mean() == 0
        plt.figure(3)
        sns.set(style="darkgrid", font_scale=1.5)
        sns.lineplot(x=np.concatenate((episode1, episode2)), y=np.concatenate((rewards1, rewards2)), label='line1')
        sns.lineplot(x=np.concatenate((episode3, episode4)), y=np.concatenate((rewards1, rewards3)), label='line2')
        plt.xlabel('step')
        plt.ylabel('reward_rate')
        plt.title('test_pic')
        plt.savefig('./pic_compare_diffpic.jpg')
        I1 = Image.open('./pic.jpg')
        I2 = Image.open('./pic_compare_diffpic.jpg')
        I1_array = np.array(I1)
        I2_array = np.array(I2)
        assert (I1_array - I2_array).mean() < 100
