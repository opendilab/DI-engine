import zmq
from threading import Thread
from collections import deque
from multiprocessing import Lock
import time


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
    def __init__(self, *args, queue_size=None, HWM=10, time_interval=10, **kwargs):
        super(ManagerZmq, self).__init__(*args, **kwargs)
        self.queue = deque(maxlen=queue_size)

        self.sender_context = zmq.Context()
        self.receiver_context = zmq.Context()
        self.request_context = zmq.Context()
        self.reply_context = zmq.Context()

        self.lock = Lock()
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
        self.lock = Lock()
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

    def receive_data(self, context, port, test_speed=False):
        receiver = context.socket(zmq.PULL)
        receiver.setsockopt(zmq.RCVHWM, self.HWM)
        receiver.setsockopt(zmq.SNDHWM, self.HWM)
        receiver.bind("tcp://*:{}".format(port))
        if test_speed:
            count = 1
            while True:
                t1 = time.time()
                data = receiver.recv_pyobj()
                t2 = time.time()
                print('count {} receiver time {}'.format(count, t2-t1))
                self._acquire_lock(self.lock)
                self.queue.append(data)
                self._release_lock(self.lock)
                t3 = time.time()
                print('count {} append time {}'.format(count, t3-t2))
                count += 1
        else:
            while True:
                data = receiver.recv_pyobj()
                self._acquire_lock(self.lock)
                self.queue.append(data)
                self._release_lock(self.lock)
                self.receive_data_count += 1
                print('({})receive pyobj {}'.format(self.name, self.receive_data_count))

    def send_data(self, context, ip, port):
        sender = context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, self.HWM)
        sender.setsockopt(zmq.RCVHWM, self.HWM)
        sender.connect("tcp://{}:{}".format(ip, port))
        while True:
            while len(self.queue) == 0:  # Note: single thread to send data
                pass
            self._acquire_lock(self.lock)
            data = self.queue.popleft()
            self._release_lock(self.lock)
            sender.send_pyobj(data)
            self.send_data_count += 1
            print('({})send pyobj {}'.format(self.name, self.send_data_count))

    def request_data(self, context, ip, port, data_type=dict):
        request = context.socket(zmq.REQ)
        request.connect("tcp://{}:{}".format(ip, port))
        print(self.name, "tcp://{}:{}".format(ip, port))
        while True:
            time.sleep(self.time_interval)
            request.send_string("request model")
            print('({})request send'.format(self.name))
            data = request.recv_pyobj()
            print('({})request recv'.format(self.name))
            self._acquire_lock(self.lock)
            assert(isinstance(data, data_type))
            self.state_dict = data
            self._release_lock(self.lock)

    def reply_data(self, context, port, req_content="request model"):
        reply = context.socket(zmq.REP)
        reply.bind("tcp://*:{}".format(port))
        print(self.name, "tcp://*:{}".format(port))
        while self.state_dict is None:
            pass
        while True:
            msg = reply.recv_string()
            print('({})reply recv'.format(self.name))
            assert(msg == req_content)
            self._acquire_lock(self.lock)
            reply.send_pyobj(self.state_dict)
            self._release_lock(self.lock)
            print('({})reply send'.format(self.name))
