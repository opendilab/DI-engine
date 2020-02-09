from collections import deque
import os
import time
import torch
import logging
import random
from multiprocessing import Lock


logger = logging.getLogger('default_logger')


class OnlineDataset(object):
    def __init__(self, data_maxlen, transform, block_data, min_update_count=0):
        # TODO container optimization
        '''
            deque
            head and tail append and pop
            randaom access
            fixed queue maxlen
            remove elements by indexes
        '''
        self.data_queue = deque(maxlen=data_maxlen)
        self.transform = transform
        self.data_maxlen = data_maxlen

        self.lock = Lock()  # TODO review lock usage
        self.push_count = 0
        self.block_data = block_data
        self.min_update_count = min_update_count

    def _acquire_lock(self):
        self.lock.acquire()

    def _release_lock(self):
        self.lock.release()

    def push_data(self, data):
        assert(isinstance(data, dict))
        self._acquire_lock()
        data['use_count'] = 0
        self.data_queue.append(data)
        self.push_count += 1
        self._release_lock()

    def _add_usage_count(self, usage_list):
        for idx in usage_list:
            self.data_queue[idx]['use_count'] += 1

    def get_sample_batch(self, batch_size, cur_model_index):
        use_block_data = self.block_data.status
        reuse_threshold = self.block_data.reuse_threshold
        staleness_threshold = self.block_data.staleness_threshold
        sleep_time = self.block_data.sleep_time
        while True:
            self._acquire_lock()
            for _ in range(self.min_update_count):
                if len(self.data_queue) > 0:
                    self.data_queue.popleft()
            if use_block_data:
                new_data_queue = deque(maxlen=self.data_maxlen)
                for item in self.data_queue:
                    if cur_model_index - item['model_index'] >= int(staleness_threshold):
                        item = None
                    if item is not None and item['use_count'] >= int(reuse_threshold):
                        item = None
                    if item is not None:
                        new_data_queue.append(item)
                self.data_queue = new_data_queue
            while not self.is_full():
                print("Blocking...wait for enough data: current({})/target({}\tpush_count({}))".format(
                      len(self.data_queue), self.data_maxlen, self.push_count))
                self._release_lock()
                time.sleep(sleep_time)
                continue
            indice = random.sample(list(range(self.data_maxlen)), batch_size)
            data = [self.data_queue[i] for i in indice]
            self._add_usage_count(indice)
            # all statistics is for drawn samples
            model_index = [d['model_index'] for d in data]
            usage = [d['use_count'] for d in data]
            data = [self.transform(d) for d in data]
            avg_model_index = sum(model_index) / len(model_index)
            avg_usage = sum(usage) / len(usage)
            push_count = self.push_count
            self.push_count = 0
            self._release_lock()
            return data, avg_usage, push_count, avg_model_index

    def extend_data(self, data_list):
        self._acquire_lock()
        for item in data_list:
            item['use_count'] = 0
        self.data_queue.extend(data_list)
        self._release_lock()

    def is_full(self):
        return len(self.data_queue) == self.data_maxlen

    def format_len(self):
        return 'current episode_infos len:{}/ready episode_infos len:{}'.format(
                    len(self.data_queue), self.data_maxlen
                )

    def __len__(self):
        return len(self.data_queue)

    @property
    def maxlen(self):
        return self.data_maxlen

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

    def load_state_dict(self, checkpoint):
        assert(isinstance(checkpoint, list))
        self._acquire_lock()
        self.data_queue.extend(checkpoint)
        self._release_lock()

    def state_dict(self):
        return list(self.data_queue)
