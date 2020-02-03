from queue import Queue
from threading import Thread
import zmq
from sc2learner.utils import build_checkpoint_helper, build_time_helper, send_array, dict2nparray, get_ip, get_pid


class BaseActor(object):

    def __init__(self, env, model, cfg=None, enable_push=True):
        assert(cfg is not None)
        self.cfg = cfg
        self.env = env
        self.model = model
        self.unroll_length = cfg.train.unroll_length

        port = cfg.communication.port
        ip = cfg.communication.ip
        push_ip = ip['actor_manager']
        push_port = port['actor_manager']
        req_ip = ip['actor_manager']
        req_port = port['actor_model']
        self.HWM = cfg.communication.HWM['actor']
        self.time_helper = build_time_helper(wrapper_type='time')

        self.zmq_context = zmq.Context()
        self.model_requestor = self.zmq_context.socket(zmq.DEALER)
        self.model_requestor.connect("tcp://%s:%s" % (req_ip, req_port))
        self.model_requestor.setsockopt(zmq.RCVTIMEO, 1000*10)
        print("tcp://%s:%s" % (req_ip, req_port))

        if enable_push:
            self.data_queue = Queue(cfg.train.actor_data_queue_size)
            self.push_thread = Thread(target=self._push_data, args=(self.zmq_context,
                                      push_ip, push_port, self.data_queue))
        self.enable_push = enable_push
        self.checkpoint_helper = build_checkpoint_helper(cfg)
        if cfg.common.load_path != '':
            self.checkpoint_helper.load(cfg.common.load_path, self.model, logger_prefix='(actor)')
        self.actor_id = '{}+{}'.format(get_ip(), get_pid())

        self._init()

    def run(self):
        self.push_thread.start()
        while True:
            self.time_helper.start_time()
            self._update_model()
            model_time = self.time_helper.end_time()
            self.time_helper.start_time()
            unroll = self._nstep_rollout()
            unroll['actor_id'] = self.actor_id
            unroll['model_index'] = self.model_index
            unroll['update_model_time'] = model_time
            data_time = self.time_helper.end_time()
            unroll['data_rollout_time'] = data_time
            print('update model time({})\tdata rollout time({})\tmodel_index({})'.format(
                model_time, data_time, self.model_index))
            if self.enable_push:
                if self.data_queue.full():
                    print('full')  # TODO warning(queue is full)
                self.data_queue.put(unroll)

    def _push_data(self, zmq_context, ip, port, queue):
        sender = zmq_context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, self.HWM)
        sender.setsockopt(zmq.RCVHWM, self.HWM)
        sender.connect("tcp://%s:%s" % (ip, port))
        while True:
            data = queue.get()
            sender.send_pyobj(data)

    def _update_model(self):
        while True:
            self.model_requestor.send_string("request model")
            try:
                state_dict = self.model_requestor.recv_pyobj()
            except zmq.error.Again:
                continue
            else:
                break
        self.model.load_state_dict(state_dict['state_dict'])
        self.model_index = state_dict['model_index']

    def _init(self):
        raise NotImplementedError

    def _nstep_rollout(self):
        raise NotImplementedError
