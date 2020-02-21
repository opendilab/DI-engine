import zmq
from threading import Thread
from collections import deque
from multiprocessing import Lock
import time
import numpy as np
import torch


def send_array(socket, array, flags=0, copy=True, track=False):
    assert(isinstance(array, np.ndarray))
    md = dict(
        dtype=str(array.dtype),
        shape=array.shape,
    )
    socket.send_json(md, flags or zmq.SNDMORE)
    return socket.send(array, flags, copy=copy, track=track)


def recv_array(socket, flags=0, copy=True, track=False):
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    array = np.frombuffer(buf, dtype=md['dtype'])
    return array.reshape(md['shape'])


def dict2nparray(data):
    result = []
    json = {}
    B = data['obs'].shape[0]
    for k, v in data.items():
        if k == 'state':
            if v is None:
                json['state'] = None
            else:
                raise NotImplementedError
        elif k == 'episode_infos' or k == 'model_index':
            json[k] = v
        else:
            L = len(v.shape)
            if L == 1:
                item = v.unsqueeze(1).numpy()
                result.append(item)
                json[k] = {'shape': item.shape, 'ori_shape': (B,)}
            elif L == 2:
                item = v.numpy()
                result.append(item)
                json[k] = {'shape': item.shape}
            else:
                raise ValueError('invalid dimension num {}'.format(L))

    return np.concatenate(result, axis=1), json


def nparray2dict(array, json):
    data = {}
    dims = []
    names = []
    for k, v in json.items():
        if k == 'state':
            data[k] = v
        elif k == 'episode_infos' or k == 'model_index':
            data[k] = v
        else:
            dims.append(v['shape'])
            names.append(k)

    sum_dims = []
    sums = 0
    for i in range(len(dims)):
        sums += dims[i]
        if i < len(dims) - 1:
            sum_dims.append(sums)
    split_array = np.split(array, sum_dims, axis=1)
    for n, a in zip(names, split_array):
        if 'ori_shape' in json[n].keys():
            a = np.reshape(a, *json[n]['ori_shape'])
        data[n] = torch.FloatTensor(a)
    return data


class ManagerBase(object):
    # Note: mainly support for multi receiver and single sender
    def __init__(self, ip, port, name):
        assert(isinstance(ip, dict))
        assert(isinstance(port, dict))
        # TODO validate whether port is available
        self.ip = ip
        self.port = port
        self.name = name

    def receive_data(self):
        raise NotImplementedError

    def send_data(self):
        raise NotImplementedError


class ManagerZmq(ManagerBase):
    def __init__(self, *args, send_queue_size=None, receive_queue_size=None, HWM=10, time_interval=10, **kwargs):
        super(ManagerZmq, self).__init__(*args, **kwargs)
        self.receive_queue_size = receive_queue_size
        self.send_queue_size = send_queue_size
        # received data is packed into a list of receive_queue_size before sent
        self.receive_queue = deque(maxlen=receive_queue_size)
        self.send_queue = deque(maxlen=send_queue_size)  # no impact to staleness

        self.sender_context = zmq.Context()
        self.receiver_context = zmq.Context()
        self.request_context = zmq.Context()
        self.reply_context = zmq.Context()

        self.send_lock = Lock()
        self.model_lock = Lock()
        self.HWM = HWM
        self.sender_thread = Thread(target=self.send_data,
                                    args=(self.sender_context, self.ip['send'], self.port['send']))
        self.receiver_thread = Thread(target=self.receive_data,
                                      args=(self.receiver_context, self.port['receive']))
        self.request_thread = Thread(target=self.request_data,
                                     args=(self.request_context, self.ip['send'], self.port['request']))
        self.reply_thread = Thread(target=self.reply_data,
                                   args=(self.reply_context, self.port['reply']))
        self.send_data_count = 0
        self.receive_data_count = 0
        self.time_interval = time_interval
        self.state_dict = None

    def _acquire_lock(self, lock):
        lock.acquire()

    def _release_lock(self, lock):
        lock.release()

    def run(self, state):
        assert(isinstance(state, dict))
        if state['sender']:
            self.sender_thread.start()
        if state['receiver']:
            self.receiver_thread.start()
        if state['forward_request']:
            self.request_thread.start()
        if state['forward_reply']:
            self.reply_thread.start()

    def receive_data(self, context, port, test_speed=True):
        receiver = context.socket(zmq.PULL)
        receiver.setsockopt(zmq.RCVHWM, self.HWM)
        receiver.setsockopt(zmq.SNDHWM, self.HWM)
        receiver.bind("tcp://*:{}".format(port))
        if test_speed:
            while True:
                t1 = time.time()
                data = receiver.recv()
                t2 = time.time()
                print('({})receive pyobj {} receiver time {}'.format(self.name, self.receive_data_count, t2-t1))
                if isinstance(data, list):
                    self.receive_queue.extend(data)
                    self.receive_data_count += len(data)
                else:
                    self.receive_queue.append(data)
                    self.receive_data_count += 1
                if len(self.receive_queue) == self.receive_queue_size:
                    self._acquire_lock(self.send_lock)
                    self.send_queue.extend(list(self.receive_queue))
                    self._release_lock(self.send_lock)
                    if len(self.send_queue) == self.send_queue_size:
                        print('Warning: Send queue full')
                    self.receive_queue.clear()
                t3 = time.time()
                print('({})receive pyobj {} append time {}'.format(self.name, self.receive_data_count, t3-t2))

        else:
            while True:
                data = receiver.recv()
                if isinstance(data, list):
                    self.receive_queue.extend(data)
                    self.receive_data_count += len(data)
                else:
                    self.receive_queue.append(data)
                    self.receive_data_count += 1
                if len(self.receive_queue) == self.receive_queue_size:
                    self._acquire_lock(self.send_lock)
                    self.send_queue.extend(list(self.receive_queue))
                    self._release_lock(self.send_lock)
                    self.receive_queue.clear()
                print('({})receive pyobj {}'.format(self.name, self.receive_data_count))

    def send_data(self, context, ip, port):
        sender = context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, self.HWM)
        sender.setsockopt(zmq.RCVHWM, self.HWM)
        sender.connect("tcp://{}:{}".format(ip, port))
        while True:
            while len(self.send_queue) == 0:  # Note: single thread to send data
                pass
            self._acquire_lock(self.send_lock)
            data = self.send_queue.popleft()
            self._release_lock(self.send_lock)
            self.send_data_count += 1
            t1 = time.time()
            sender.send(data)
            t2 = time.time()
            print('({})send {} time {}'.format(self.name, self.send_data_count, t2-t1))

    def request_data(self, context, ip, port):
        request = context.socket(zmq.DEALER)
        request.setsockopt(zmq.RCVTIMEO, 1000*10)
        request.connect("tcp://{}:{}".format(ip, port))
        print(self.name, "tcp://{}:{}".format(ip, port))
        while True:
            time.sleep(self.time_interval)
            while True:
                request.send_string("request model")
                try:
                    data = request.recv()
                except zmq.error.Again:
                    continue
                else:
                    print('({})update state_dict'.format(self.name))
                    break
            self._acquire_lock(self.model_lock)
            self.state_dict = data
            self._release_lock(self.model_lock)

    def reply_data(self, context, port, req_content="request model"):
        reply = context.socket(zmq.DEALER)  # TODO why REP can't receive the message from DEALER
        reply.bind("tcp://*:{}".format(port))
        print(self.name, "tcp://*:{}".format(port))
        while self.state_dict is None:
            pass
        while True:
            msg = reply.recv_string()
            assert(msg == req_content)
            self._acquire_lock(self.model_lock)
            reply.send(self.state_dict)
            self._release_lock(self.model_lock)
            print('({})reply model'.format(self.name))
