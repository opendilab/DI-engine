from threading import Thread
from multiprocessing import Lock
from collections import deque
import random
import time
import zmq
import pickle
import os


class JobManager():
    """
    Object maintaining a list of current running task
    Check timed-out jobs and return them to queue
    Fields for job dict:
        actor_id
        job_id
        start_time: only used in the manager
        last_checkin: only in the manager
        step: only used in the manager for stat and checkpointing
        start_rollout_at: For actors, don't send rollout to learner before this, but should always check in
        game_vs_bot:
            seed: the enviroment and action choice seed
            difficulty
    """

    def __init__(self, cfg, job_generator):
        self.job_generator = job_generator
        self.check_in_timeout = cfg.job_manager.check_in_timeout
        self.discard_timeout_jobs = cfg.job_manager.discard_timeout_jobs
        self.check_job_check_in_to_thread = Thread(
            target=self.check_job_check_in_timeout)
        self.check_job_check_in_to_thread.start()
        self.running_job_pool = {}
        self.job_pool_lock = Lock()
        self.available_job_queue = deque()
        self.last_request_req_id = {}
        self.cached_response = {}

    def get_job(self, actor_id, req_id):
        if actor_id in self.last_request_req_id:
            if self.last_request_req_id[actor_id] == req_id\
                    and actor_id in self.cached_response:
                # noticed this is a repeated request, send cached job
                print('WARNING: received repeated request from {}'.format(actor_id))
                job = self.cached_response[actor_id]
                job['start_time'] = time.time()
                job['last_checkin'] = job['start_time']
                self.job_pool_lock.acquire()
                self.running_job_pool[job['job_id']] = job
                self.job_pool_lock.release()
                return job
            elif self.last_request_req_id[actor_id] > req_id:
                print('WARNING: received older request from {}, actor restarted?'.format(actor_id))
        self.last_request_req_id[actor_id] = req_id

        if self.available_job_queue:
            job = self.available_job_queue.pop_left()
        else:
            job = self.job_generator.gen()
        job['actor_id'] = actor_id
        job['start_time'] = time.time()
        job['last_checkin'] = job['start_time']
        job['step'] = 0
        self.job_pool_lock.acquire()
        self.running_job_pool[job['job_id']] = job
        self.job_pool_lock.release()
        self.cached_response[actor_id] = job
        return job

    def job_check_in_callback(self, check_in_message):
        """
        Expected check-in message fields:
            job_id
            done
            step
            actor_id
        """
        self.job_pool_lock.acquire()
        actor_id = check_in_message['actor_id']
        job_id = check_in_message['job_id']
        if check_in_message['done']:
            self.job_generator.job_done_callback(check_in_message)
            if job_id in self.running_job_pool:
                del self.running_job_pool[job_id]
            else:
                print('WARNING: received check in for non-existing job {} from {}'
                      .format(job_id, actor_id))
            ret = {'type': 'ack', 'actor_id': actor_id}
        else:
            if job_id in self.running_job_pool:
                self.running_job_pool[job_id]['last_checkin'] = time.time()
                self.running_job_pool[job_id]['step'] = check_in_message['step']
                ret = {'type': 'ack', 'actor_id': actor_id}
            else:
                print('WARNING: received check in for non-existing job {} from {}'
                      .format(job_id, actor_id))
                ret = {'type': 'job_cancel',
                       'job_id': job_id, 'actor_id': actor_id}
        self.job_pool_lock.release()
        return ret

    def check_job_check_in_timeout(self):
        # waiting for starting actors and other managers
        time.sleep(1.5 * self.check_in_timeout)
        while True:
            expired_jobs = []
            self.job_pool_lock.acquire()
            for job_id, job in self.running_job_pool.items():
                dt = time.time() - job['last_checkin']
                if dt > self.check_in_timeout:
                    expired_jobs.append(job_id)
                    print('Job {} assigned to {} expired, {} from last check in'
                          .format(job_id, job['actor_id'], dt))
            for job_id in expired_jobs:
                job = self.running_job_pool[job_id]
                del self.running_job_pool[job_id]
                if not self.discard_timeout_jobs:
                    job['actor_id'] = None
                    job['start_time'] = None
                    job['last_checkin'] = None
                    job['start_rollout_at'] = job['step']
                    self.available_job_queue.append(job)
            self.job_pool_lock.release()
            time.sleep(10)  # check timeout per 10sec

    def get_checkpoint_data(self):
        """
        Pushing every recorded job in running pool to the queue.
        When loading, the jobs will be restored to the queue and
        have higher priority than generating new jobs
        """
        checkpoint_joblist = []
        for job in self.available_job_queue:
            checkpoint_joblist.append(job)
        for job_id, job in self.running_job_pool:
            job['actor_id'] = None
            job['start_time'] = None
            job['last_checkin'] = None
            job['start_rollout_at'] = job['step']
            checkpoint_joblist.append(job)
        return checkpoint_joblist

    def load_checkpoint_data(self, data):
        self.available_job_queue = deque(data)


class JobGenerator():
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg.job_manager.seed)
        self.next_job_id = 0

    def gen(self):
        job = {}
        difficulty = random.choice(self.cfg.env.bot_difficulties.split(','))
        seed = random.randint(0, 2**32 - 1)
        job['game_vs_bot'] = {'seed': seed,
                              'difficulty': difficulty}
        print("New Job {}: Game&Pytorch Seed: {} Difficulty: {}".format(
            self.next_job_id, seed, difficulty))
        job['job_id'] = self.next_job_id
        job['start_rollout_at'] = 0
        self.next_job_id += 1
        return job

    def job_done_callback(self, last_checkin_message):
        # dummy
        print('Job {} done by {}'
              .format(last_checkin_message['job_id'], last_checkin_message['actor_id']))

    def get_checkpoint_data(self):
        return {'random_state': random.getstate(),
                'next_job_id': self.next_job_id}

    def load_checkpoint_data(self, data):
        random.setstate(data['random_state'])
        self.next_job_id = data['next_job_id']


class Coordinator():
    def __init__(self, cfg, port):
        self.cfg = cfg
        self.context = zmq.Context()
        self.job_generator = JobGenerator(cfg)
        self.job_manager = JobManager(cfg, self.job_generator)
        self.coordinator_thread = Thread(target=self.coordinator,
                                         args=(port,))
        self.save_path = os.path.join(cfg.common.save_path, 'checkpoints')
        self.max_model_index = 0
        self.last_check_point = 0
        if self.cfg.common.load_path != '':
            with open(self.cfg.common.load_path, 'rb') as lf:
                dat = pickle.load(lf)
            self.load_checkpoint_data(dat)

    def run(self):
        self.coordinator_thread.start()

    def coordinator(self, port):
        """
        Fields in datagram dict sent by actors
        common:
            type: "check in" or "job req"
            actor_id
            model_index
        check in only:
            job_id
            done: True or False
            step
        job request only:
            req_id: an serial number to recognize repeated requests

        Fields in returned data
        type: job, ack, job_cancel
        actor_id
        for job:
            job: job dict
        for job_cancel:
            actor_id
            job_id
        """
        self.connector = self.context.socket(zmq.ROUTER)
        self.connector.bind("tcp://*:{}".format(port))
        print("Bind to tcp://*:{}".format(port))
        while True:
            ident, data = self.connector.recv_multipart()
            data = pickle.loads(data)
            assert(isinstance(data, dict))
            if 'model_index' in data:
                if data['model_index'] > self.max_model_index:
                    self.max_model_index = data['model_index']
            if self.max_model_index > self.last_check_point\
                    and self.max_model_index % self.cfg.logger.save_freq == 0:
                self.last_checkpoint = self.max_model_index
                self.save_checkpoint()
            if data['type'] == 'check in':
                ret = self.job_manager.job_check_in_callback(data)
                self.connector.send_multipart([ident, pickle.dumps(ret)])
                continue
            elif data['type'] == 'job req':
                job = self.job_manager.get_job(
                    data['actor_id'], data['req_id'])
                ret = {'type': 'job', 'job': job, 'actor_id': data['actor_id']}
                self.connector.send_multipart([ident, pickle.dumps(ret)])
                continue
            else:
                print('ERROR: Unknown request type {}'.format(data['type']))

    def save_checkpoint(self):
        print('Saving checkpoint {}'.format(self.max_model_index))
        cd = self.get_checkpoint_data()
        path = os.path.join(
            self.savepath, 'coordinator_iter{}.pickle'.format(self.max_model_index))
        with open(path, 'wb') as of:
            pickle.dump(cd, of)

    def get_checkpoint_data(self):
        jg = self.job_generator.get_checkpoint_data()
        jm = self.job_manager.get_checkpoint_data()
        return {'job_manager': jm, 'job_generator': jg}

    def load_checkpoint_data(self, data):
        assert(isinstance(data, dict))
        self.job_generator.load_checkpoint_data(data['job_generator'])
        self.job_manager.load_checkpoint_data(data['job_manager'])
