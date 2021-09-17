import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def plot(data: list[dict], xlabel: str, ylabel: str, title: str):
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
        for k,v in nowdata.items():
            if k=='step':
                step = v
            if k=='value':
                value = v
            if k=='label':
                label = v
        sns.lineplot(x=step, y=value, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()