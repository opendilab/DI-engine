import os
import csv
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import str


def plot(data: list, xlabel: str, ylabel: str, title: str, pth: str = './picture.jpg'):
    """
    Overview:
        Draw training polyline
    Interface:
        data (:obj:`List[Dict]`): the data we will use to draw polylines
            data[i]['step']: horizontal axis data
            data[i]['value']: vertical axis data
            data[i]['label']: the data label
        xlabel (:obj:`str`): the x label name
        ylabel (:obj:`str`): the y label name
        title (:obj:`str`): the title name
    """
    sns.set(style="darkgrid", font_scale=1.5)
    for nowdata in data:
        step, value, label = nowdata['x'], nowdata['y'], nowdata['label']
        sns.lineplot(x=step, y=value, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(pth)


def plotter(root: str, titles: str, labels: str, x_axes: str, y_axes: str):
    '''
    root: the location of file containing all algorithms. Each is a file containing\
         five seeds of event file generated from tensorboard
    titles: the titles to be plotted in each diagram
    labels: the labels for each algorithm
    x_axes: the x-axis for each diagram
    y_axes: the y-axis for each diagram
    '''

    headers = ['steps', 'value']
    root = '/mnt/lustre/nieyunpeng/benchmark_draw/pong_dqn'

    count_file = 0
    for root, dirs, _ in os.walk(root):
        for d in dirs:  # ToDo: plot together
            title = titles[count_file]
            label = labels[count_file]
            x_axis = x_axes[count_file]
            y_axis = y_axes[count_file]
            count_file += 1
            exp_path = os.path.join(root, d)
            env, agent = d.split('_')
            print(env, agent)
            results = {}
            for exp_root, _, exp_files in os.walk(exp_path):
                reward_seeds = []
                for exp_i, exp_file in enumerate(exp_files):  # ToDo: offline
                    print(exp_i)
                    ea = event_accumulator.EventAccumulator(os.path.join(exp_root, exp_file))
                    ea.Reload()
                    rewards = ea.scalars.Items('evaluator_step/reward_mean')
                    reward_seeds.append(rewards)
                dummy = [len(i) for i in reward_seeds]
                max_steps = max([len(i) for i in reward_seeds])
                index = dummy.index(max([len(i) for i in reward_seeds]))
                result = []
                for i, reward in enumerate(reward_seeds[index]):
                    result.append(reward.step)
                results['step'] = result
                for j in range(len(reward_seeds)):
                    reward_j = []
                    for i, reward in enumerate(reward_seeds[j]):
                        reward_j.append(reward.value)
                    while i < max_steps - 1:
                        reward_j.append(reward.value)
                        i += 1
                    i = 0
                    results[j] = reward_j
            steps = results['step'] * (len(results) - 1)
            results.pop('step')
            value = []
            for i in range(len(results)):
                value.extend(results[i])

            sns.lineplot(x=steps, y=value, label=label, color='#ad1457')
            sns.set(style="darkgrid", font_scale=1.5)
            plt.title(title)
            plt.legend(loc='upper left', prop={'size': 8})
            plt.xlabel(x_axis, fontsize=15)
            plt.ylabel(y_axis, fontsize=15)
            plt.show()

            csv_dicts = []
            for i, _ in enumerate(steps):
                csv_dicts.append({'steps': steps[i], 'value': value[i]})
            with open(os.path.join(exp_path, '{}.csv'.format(d)), 'w', newline='') as f:
                writer = csv.DictWriter(f, headers)
                writer.writeheader()
                writer.writerows(csv_dicts)
