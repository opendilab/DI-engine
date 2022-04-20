import os
import csv
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


def plot(data: list, xlabel: str, ylabel: str, title: str, pth: str = './picture.png'):
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
    plt.legend(loc='upper left', prop={'size': 8})
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title)
    plt.savefig(pth, bbox_inches='tight')


def plotter(
    root: str,
    titles: List[str],
    labels: List[str],
    x_axes: List[str],
    y_axes: List[str],
    plot_together: bool = False,
    plot_together_x_axis: str = None,
    plot_together_y_axis: str = None,
    plot_together_title: str = None
):
    '''
    root: the location of the folder containing all algorithms data. Each is a folder containing a few seeds of event file generated from TensorBoard
    titles: the titles to be plotted in each diagram; This has no effect in plot_together mode
    labels: the labels for each algorithm
    x_axes: the x-axis for each diagram; This has no effect in plot_together mode
    y_axes: the y-axis for each diagram; This has no effect in plot_together mode
    plot_together: whether to plot together or not
    plot_together_x_axis: if plot_together, indicates the x axis for the plot
    plot_together_y_axis: if plot_together, indicates the y axis for the plot
    plot_together_title: if plot_together, indicates the title for the plot
    '''

    headers = ['steps', 'value']

    count_file = 0
    data_holder = []  # for plotting together only
    foot_root = root  # for plotting together only
    for root, dirs, _ in os.walk(root):
        if len(dirs) > 1:
            dirs.sort()
        for d in dirs:

            title = titles[count_file]
            label = labels[count_file]
            x_axis = x_axes[count_file]
            y_axis = y_axes[count_file]
            count_file += 1
            exp_path = os.path.join(root, d)
            print(exp_path)
            env, agent = d.split('_')
            print(env, agent)
            results = {}
            for exp_root, _, exp_files in os.walk(exp_path):
                reward_seeds = []
                for exp_i, exp_file in enumerate(exp_files):  # ToDo: offline
                    if ('.csv' in exp_file or '.png' in exp_file):
                        continue
                    try:
                        ea = event_accumulator.EventAccumulator(os.path.join(exp_root, exp_file))
                        ea.Reload()
                        rewards = ea.scalars.Items('evaluator_step/reward_mean')
                        reward_seeds.append(rewards)
                    except:
                        raise Exception("{0} should not be in the directory: {1}".format(exp_file, exp_path))
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
            if not plot_together:
                figure_lineplot = sns.lineplot(x=steps, y=value, label=label, color='#ad1457')
                #sns.set(style="darkgrid", font_scale=1.5)
                plt.title(title)
                plt.legend(loc='upper left', prop={'size': 8})
                plt.xlabel(x_axis, fontsize=15)
                plt.ylabel(y_axis, fontsize=15)
                #plt.show()
                figure = figure_lineplot.get_figure()
                figure.savefig(exp_path + '/' + title + '.png')
                plt.close()
            else:
                data_holder.append({'x': steps, 'y': value, 'label': label})
            csv_dicts = []
            for i, _ in enumerate(steps):
                csv_dicts.append({'steps': steps[i], 'value': value[i]})
            with open(os.path.join(exp_path, '{}.csv'.format(d)), 'w', newline='') as f:
                writer = csv.DictWriter(f, headers)
                writer.writeheader()
                writer.writerows(csv_dicts)
    if plot_together:
        assert type(plot_together_x_axis) is str and type(plot_together_y_axis) is str and type(
            plot_together_title
        ) is str, 'Please indicate the x-axis, the y-axis and the title'
        plot(
            data_holder,
            plot_together_x_axis,
            plot_together_y_axis,
            plot_together_title,
            pth=foot_root + foot_root[foot_root.rfind('/'):] + '.png'
        )
