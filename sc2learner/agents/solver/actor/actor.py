from queue import Queue
from threading import Thread
import zmq
import os
from sc2learner.utils import build_checkpoint_helper, build_time_helper, send_array, dict2nparray, get_pid
import time


class BaseActor(object):

    def __init__(self, cfg, model=None, enable_push=True):
        assert(cfg is not None)
        self.cfg = cfg
        self.unroll_length = cfg.train.unroll_length
        self.env = None
        self.model = None  # will be created in self._init

        port = cfg.communication.port
        ip = cfg.communication.ip
        if ip['actor_manager'] == 'auto':
            # IP of actor is added in train_ppo.py
            prefix = '.'.join(ip.actor.split('.')[:3])
            ip['actor_manager'] = ip.manager_node[prefix]
        push_ip = ip['actor_manager']
        push_port = port['actor_manager']
        req_ip = ip['actor_manager']
        req_port = port['actor_model']
        self.HWM = cfg.communication.HWM['actor']
        self.time_helper = build_time_helper(wrapper_type='time')

        self.zmq_context = zmq.Context()
        self.model_requestor = self.zmq_context.socket(zmq.DEALER)
        self.model_requestor.connect("tcp://%s:%s" % (req_ip, req_port))
        self.model_requestor.setsockopt(zmq.RCVTIMEO, 1000*15)
        # force ZMQ keep only the most recent model received
        # avoid high staleness after network unstablity
        # require zmq 4.x
        self.model_requestor.setsockopt(zmq.CONFLATE, 1)
        print("model_requestor: tcp://%s:%s" % (req_ip, req_port))

        self.job_requestor = self.zmq_context.socket(zmq.DEALER)
        self.job_requestor.connect("tcp://{}:{}"
                                   .format(ip['actor_manager'], port['coordinator_relayed']))
        self.job_requestor.setsockopt(zmq.RCVTIMEO, 1000*10)
        print(
            "job_requestor: tcp://{}:{}".format(ip['actor_manager'], port['coordinator_relayed']))
        self.job_request_id = 0

        if enable_push:
            self.data_queue = Queue(cfg.train.actor_data_queue_size)
            self.push_thread = Thread(target=self._push_data, args=(self.zmq_context,
                                                                    push_ip, push_port, self.data_queue))
        self.enable_push = enable_push
        # self.checkpoint_helper = build_checkpoint_helper(cfg)
        # if cfg.common.load_path != '':
        #     self.checkpoint_helper.load(
        #         cfg.common.load_path, self.model, logger_prefix='(actor)')
        # if SLURM_JOB_ID is not available, failback to 'PID'+pid
        self.actor_id = '{}+{}'.format(ip.actor,
                                       os.getenv('SLURM_JOB_ID', 'PID'+str(get_pid())))
        self.job_id = None
        self.job_cancelled = False
        self.start_rollout_at = 0
        self.model_index = 0
        self._init()

    def run(self):
        self.push_thread.start()
        while True:
            self.time_helper.start_time()
            self._update_model()  # should be blocking
            model_time = self.time_helper.end_time()
            self.time_helper.start_time()
            unroll = self._nstep_rollout()
            unroll['actor_id'] = self.actor_id
            unroll['model_index'] = self.model_index
            unroll['update_model_time'] = model_time
            data_time = self.time_helper.end_time()
            unroll['data_rollout_time'] = data_time
            # this should be incremented in _nstep_rollout
            unroll['step'] = self.step
            print('update model time({})\tdata rollout time({})\tmodel_index({})'.format(
                model_time, data_time, self.model_index))
            if self.enable_push and self.step >= self.start_rollout_at:
                if self.data_queue.full():
                    print('WARNING: Actor send queue full')
                self.data_queue.put(unroll)
            # checking in
            self._do_check_in()
            if self.done or self.job_cancelled:
                self.job_cancelled = False
                self.done = False
                self._init()

    def _push_data(self, zmq_context, ip, port, queue):
        sender = zmq_context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDHWM, self.HWM)
        sender.setsockopt(zmq.RCVHWM, self.HWM)
        sender.connect("tcp://%s:%s" % (ip, port))
        while True:
            data = queue.get()
            sender.send_pyobj(data)

    def _update_model(self):
        self.model_requestor.setsockopt(zmq.RCVTIMEO, 1000*15)
        while True:
            self.model_requestor.send_string("request model")
            try:
                state_dict = self.model_requestor.recv_pyobj()
            except zmq.error.Again:
                print('WARNING: Model Request Timeout')
                time.sleep(1)
                continue
            else:
                break
        self.model.load_state_dict(state_dict['state_dict'])
        self.model_index = state_dict['model_index']
        self.model_age = time.time() - state_dict['timestamp']
        print('Model Wallclock Age:{}'.format(self.model_age))
        if(self.model_age > 250):  # TODO: add a entry in config file
            print('WARNING: Old Model Received, start clearing receive queue.')
            self.model_requestor.setsockopt(zmq.RCVTIMEO, 1000*3)
            while True:
                try:
                    state_dict = self.model_requestor.recv_pyobj()
                    print('Ate queued model')
                except zmq.error.Again:
                    print('Timeout')
                    break

    def _do_check_in(self):
        check_in_message = {'type': 'check in',
                            'job_id': self.job_id,
                            'done': self.done,
                            'step': self.step,
                            'actor_id': self.actor_id,
                            'model_index': self.model_index}
        try:
            self.job_requestor.send_pyobj(check_in_message)
            reply = self.job_requestor.recv_pyobj()
            assert(isinstance(reply, dict))
            if reply['type'] == 'ack':
                pass
            elif reply['type'] == 'job_cancel'\
                    and reply['actor_id'] == self.actor_id\
                    and reply['job_id'] == self.job_id:
                print('WARNING: Job Cancel Request Received')
                self.job_cancelled = True

        except zmq.error.Again:
            print('WARNING: Checkin Timeout')

    def _request_job(self):
        while True:
            job_request = {'type': 'job req',
                           'req_id': self.job_request_id,
                           'actor_id': self.actor_id,
                           'model_index': self.model_index}
            try:
                self.job_requestor.send_pyobj(job_request)
                reply = self.job_requestor.recv_pyobj()
                assert(isinstance(reply, dict))
                if(reply['type'] != 'job'):
                    print('WARNING: received unknown response for job req, type:{}'
                          .format(reply['type']))
                    continue
                if(reply['actor_id'] != self.actor_id):
                    print('WARNING: received job is assigned to another actor')
                self.job_request_id += 1
                return reply['job']
            except zmq.error.Again:
                print('WARNING: Job Request Timeout')
                continue

    def _init(self):
        raise NotImplementedError

    def _nstep_rollout(self):
        raise NotImplementedError

    def _create_env(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError
