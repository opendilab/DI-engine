from queue import Queue
from threading import Thread
import zmq
import torch
from sc2learner.utils import build_checkpoint_helper


class BaseActor(object):

    def __init__(self, env, model, cfg=None, enable_push=True):
        assert(cfg is not None)
        self.cfg = cfg
        self.env = env
        self.model = model
        self.unroll_length = cfg.train.unroll_length

        self.zmq_context = zmq.Context()
        self.model_requestor = self.zmq_context.socket(zmq.REQ)
        learner_ip = cfg.communication.learner_ip
        port = cfg.communication.port
        self.model_requestor.connect("tcp://%s:%s" % (learner_ip, port['learner']))
        if enable_push:
            self.data_queue = Queue(cfg.train.actor_data_queue_size)
            self.push_thread = Thread(target=self._push_data, args=(self.zmq_context,
                                      learner_ip, port['actor'], self.data_queue))
        self.enable_push = enable_push
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        if cfg.common.load_path != '':
            self.checkpoint_helper.load(cfg.common.load_path, self.model, logger_prefix='(actor)')

        self._init()

    def run(self):
        self.push_thread.start()
        while True:
            self._update_model()
            unroll = self._nstep_rollout()
            if self.enable_push:
                if self.data_queue.full():
                    print('full')  # TODO warning(queue is full)
                self.data_queue.put(unroll)

    def _push_data(self, zmq_context, ip, port, queue):
        sender = zmq_context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, 1)
        sender.setsockopt(zmq.RCVHWM, 1)
        sender.connect("tcp://%s:%s" % (ip, port))
        while True:
            data = queue.get()
            sender.send_pyobj(data)

    def _update_model(self):
        self.model_requestor.send_string("request model")
        state_dict = self.model_requestor.recv_pyobj()
        self.model.load_state_dict(state_dict)

    def _init(self):
        raise NotImplementedError

    def _nstep_rollout(self):
        raise NotImplementedError
