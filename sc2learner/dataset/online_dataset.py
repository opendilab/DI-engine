from collections import deque
import random
import os
import torch
import logging


logger = logging.getLogger('default_logger')


class OnlineDataset(object):
    def __init__(self, data_maxlen, episode_maxlen, transform):
        self.data_queue = deque(maxlen=data_maxlen)
        self.transform = transform
        self.data_maxlen = data_maxlen
        self.episode_maxlen = episode_maxlen
        self.episode_queue = deque(maxlen=episode_maxlen)

    def push_data(self, data):
        self.data_queue.append(data)

    def extend_data(self, data_list):
        self.data_list.extend(data_list)

    def push_episode_info(self, episode_info):
        self.episode_queue.append(episode_info)

    def is_episode_full(self):
        return len(self.episode_queue) == self.episode_maxlen

    def episode_len(self):
        return 'current episode_infos len:{}/ready episode_infos len:{}'.format(
                    len(self.episode_queue), self.episode_maxlen
                )

    def __len__(self):
        return len(self.data_queue)

    def __getitem__(self, idx):
        return self.transform(self.data_queue[idx])

    def load_data(self, data_dir, ratio=1):
        assert(ratio >= 1)
        origin_data_list = list(os.listdir(data_dir))
        if len(origin_data_list) <= self.data_maxlen:
            data_list = origin_data_list
        else:
            data_list = sorted(origin_data_list)[:self.data_maxlen]
        temp_list = []
        for idx, item in enumerate(data_list):
            temp_list.append(torch.load(os.path.join(data_dir, item)))
            if (idx + 1) % ratio == 0:
                self.extend_data(temp_list)
                self.push_episode_info(temp_list[0]['episode_info'])
                temp_list = []
        logger.info("load data in {}".format(data_dir))
