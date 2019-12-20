from collections import deque
import os
import time
import torch
import logging
from multiprocessing import Lock


logger = logging.getLogger('default_logger')


class OnlineDataset(object):
    def __init__(self, data_maxlen, transform, block_data):
        self.data_queue = deque(maxlen=data_maxlen)
        self.data_usage_count_queue = deque(maxlen=data_maxlen)
        self.transform = transform
        self.data_maxlen = data_maxlen

        self.lock = Lock()  # TODO review lock usage
        self.push_count = 0
        self.block_data = block_data

    def _acquire_lock(self):
        self.lock.acquire()

    def _release_lock(self):
        self.lock.release()

    def push_data(self, data):
        self._acquire_lock()
        self.data_queue.append(data)
        self.data_usage_count_queue.append(0)
        self.push_count += 1
        self._release_lock()

    def _add_usage_count(self, usage_list):
        for idx in usage_list:
            self.data_usage_count_queue[idx] += 1

    def get_indice_data(self, indice):
        use_block_data = self.block_data.status
        while True:
            self._acquire_lock()
            data = [self[i] for i in indice]
            model_index = [d['model_index'] for d in data]
            avg_model_index = sum(model_index) / len(model_index)
            usage = [self.data_usage_count_queue[i] for i in indice]
            avg_usage = sum(usage) / len(usage)
            if use_block_data:
                threshold = self.block_data.threshold
                sleep_time = self.block_data.sleep_time
                if avg_usage <= threshold:
                    self._add_usage_count(indice)
                    push_count = self.push_count
                    self.push_count = 0
                    self._release_lock()
                    return data, avg_usage, push_count, avg_model_index
                else:
                    self._release_lock()
                    print("Blocking...Current AVG usage: {}".format(avg_usage))
                    time.sleep(sleep_time)
            else:
                push_count = self.push_count
                self.push_count = 0
                self._add_usage_count(indice)
                self._release_lock()
                return data, avg_usage, push_count, avg_model_index

    def extend_data(self, data_list):
        self._acquire_lock()
        self.data_queue.extend(data_list)
        self.data_usage_count_queue.extend([0 for _ in range(len(data_list))])
        self._release_lock()

    def is_full(self):
        return len(self.data_queue) == self.data_maxlen

    def format_len(self):
        return 'current episode_infos len:{}/ready episode_infos len:{}'.format(
                    len(self.data_queue), self.data_maxlen
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
                temp_list = []
        logger.info("load data in {}".format(data_dir))

    def load_data_from_checkpoint(self, checkpoint):
        assert(isinstance(checkpoint, list))
        self.extend_data(checkpoint)

    def create_checkpoint(self):
        return list(self.data_queue)
