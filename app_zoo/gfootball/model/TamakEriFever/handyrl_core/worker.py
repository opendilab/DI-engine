# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# worker and gather

import os
import random
import threading
import time
import yaml
import functools
from socket import gethostname
from collections import deque
import multiprocessing as mp

import numpy as np

from .environment import prepare_env, make_env
from .connection import QueueCommunicator
from .connection import send_recv, open_multiprocessing_connections
from .connection import connect_socket_connection, accept_socket_connections
from .generation import Generator
from .replay import Replayer
from .evaluation import Evaluator, Agent
from .evaluation import RandomAgent, IdleAgent, RightAgent
from .evaluation import BuiltinAgent, RuleBasedAgentA, RuleBasedAgentB, RuleBasedAgentC
from .evaluation import RuleBasedAgentD, RuleBasedAgentE, RuleBasedAgentF
#from .evaluation import SACNN120, Hybrid373, Hybrid684, Hybrid741, Hybrid610Builtin
#from .evaluation import Hybrid700, Hybrid829, Hybrid855, HybridAllowShot893
#from .evaluation import Hybrid978, HybridAllowShot1017
#from .evaluation import HybridAllowShot1175, StickyUPGO991, StickyUPGO1550
#from .evaluation import ScoreReward1428, NewBig163
from .model import BaseModel


class Worker:
    def __init__(self, args, conn, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        self.args = args
        self.conn = conn
        self.latest_model = -1, None

        self.env = make_env({**args['env'], 'id': wid})
        self.generator = Generator(self.env, self.args)
        self.replayer = Replayer(self.env, self.args)
        self.evaluator = Evaluator(self.env, self.args)

        random.seed(args['seed'] + wid)

    def __del__(self):
        print('closed worker %d' % self.worker_id)

    def _gather_models(self, model_ids, args):
        model_pool = {}
        for model_id in model_ids:
            if model_id < 0:
                if args['role'] == 'e':
                    model_pool[model_id] = None
                else:
                    models = {
                        IdleAgent: 2,
                        RandomAgent: 1,
                        RightAgent: 1,
                        RuleBasedAgentA: 1,
                        RuleBasedAgentB: 2,
                        RuleBasedAgentC: 12,
                        #RuleBasedAgentD: 10,
                        RuleBasedAgentE: 1,
                        RuleBasedAgentF: 1,
                        BuiltinAgent: 24,
                        #SACNN120: 1,
                        #Hybrid373: 2,
                        #Hybrid684: 2,
                        #Hybrid700: 2,
                        #Hybrid741: 2,
                        #Hybrid829: 2,
                        #Hybrid855: 2,
                        #Hybrid978: 2,
                        #HybridAllowShot893: 2,
                        #HybridAllowShot1017: 2,
                        #Hybrid610Builtin: 10,
                        #HybridAllowShot1175: 5,
                        #StickyUPGO991: 2,
                        #StickyUPGO1550: 4,
                        #ScoreReward1428: 12,
                        #NewBig163: 5,
                    }
                    def normalize(w):
                        s = sum(w)
                        return [p / s for p in w]
                    model_pool[model_id] = random.choices(list(models.keys()), k=1, weights=normalize(list(models.values())))[0]()
            elif model_id not in model_pool:
                if model_id == self.latest_model[0]:
                    # use latest model
                    model_pool[model_id] = self.latest_model[1]
                else:
                    # get model from server
                    model_pool[model_id] = send_recv(self.conn, ('model', model_id))
                    # update latest model
                    if model_id > self.latest_model[0]:
                        self.latest_model = model_id, model_pool[model_id]
        return model_pool

    def run(self):
        while True:
            args = send_recv(self.conn, ('args', None))
            role = args['role']

            models = {}
            if 'model_id' in args:
                model_ids = list(args['model_id'].values())
                model_pool = self._gather_models(model_ids, args)

                # make dict of models
                for p, model_id in args['model_id'].items():
                    models[p] = model_pool[model_id]

            if role == 'g':
                opening_models = random.choices([
                    RuleBasedAgentA, RuleBasedAgentB, RuleBasedAgentC,
                    #RuleBasedAgentD,
                    RuleBasedAgentE, RuleBasedAgentF,
                ], k=2)
                opening_models = [m() for m in opening_models]

                r = random.random()
                if r < 0.4:
                    models[1 - args['player'][0]] = Agent(models[args['player'][0]])
                elif r < 0.55:
                    args['player'] = [0, 1]  # both players are trained agent
                    models[1] = models[0]

                episode = self.generator.execute(models, opening_models, args)
                send_recv(self.conn, ('episode', episode))

            elif role == 'r':
                replay = self.replayer.execute(args)
                send_recv(self.conn, ('replay', replay))

            elif role == 'e':
                result = self.evaluator.execute(models, args)
                player_model_id = args['model_id'][args['player'][0]]
                send_recv(self.conn, ('result', (player_model_id, result)))


def make_worker_args(args, n_ga, gaid, wid, conn):
    return args, conn, wid * n_ga + gaid


def open_worker(args, conn, wid):
    worker = Worker(args, conn, wid)
    worker.run()


class Gather(QueueCommunicator):
    def __init__(self, args, conn, gaid):
        print('started gather %d' % gaid)
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = deque([])
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        n_pro, n_ga = args['worker']['num_process'], args['worker']['num_gather']

        num_workers_per_gather = (n_pro // n_ga) + int(gaid < n_pro % n_ga)
        worker_conns = open_multiprocessing_connections(
            num_workers_per_gather,
            open_worker,
            functools.partial(make_worker_args, args, n_ga, gaid)
        )

        for conn in worker_conns:
            self.add(conn)

        self.args_buf_len = 1 + len(worker_conns) // 4
        self.result_buf_len = 1 + len(worker_conns) // 4

    def __del__(self):
        print('finished gather %d' % self.gather_id)

    def run(self):
        while True:
            conn, (command, args) = self.recv()
            if command == 'args':
                # When requested argsments, return buffered outputs
                if len(self.args_queue) == 0:
                    # get muptilple arguments from server and store them
                    self.server_conn.send((command, [None] * self.args_buf_len))
                    self.args_queue += self.server_conn.recv()

                next_args = self.args_queue.popleft()
                self.send(conn, next_args)

            elif command in self.data_map:
                # answer data request as soon as possible
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                # return flag first and store data
                self.send(conn, None)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.result_buf_len:
                    # send datum to server after buffering certain number of datum
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.server_conn.recv()
                    self.result_send_map = {}
                    self.result_send_cnt = 0


def gather_loop(args, conn, gaid):
    try:
        gather = Gather(args, conn, gaid)
        gather.run()
    finally:
        gather.shutdown()


class Workers(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        if self.args['remote']:
            # prepare listening connections
            def worker_server(port):
                conn_acceptor = accept_socket_connections(port=port, timeout=0.5)
                print('started worker server %d' % port)
                while not self.shutdown_flag:  # use super class's flag
                    conn = next(conn_acceptor)
                    if conn is not None:
                        self.add(conn)
                print('finished worker server')
            # use super class's thread list
            self.threads.append(threading.Thread(target=worker_server, args=(9998,)))
            self.threads[-1].start()
        else:
            # open local connections
            for i in range(self.args['worker']['num_gather']):
                conn0, conn1 = mp.connection.Pipe(duplex=True)
                mp.Process(target=gather_loop, args=(self.args, conn1, i)).start()
                conn1.close()
                self.add(conn0)


def entry(entry_args):
    conn = connect_socket_connection(entry_args['remote_host'], 9999)
    conn.send(entry_args)
    args = conn.recv()
    conn.close()
    return args


def worker_main(args):
    entry_args = args['entry_args']
    entry_args['host'] = gethostname()

    args = entry(entry_args)
    print(args)
    prepare_env(args['env'])

    # open workers
    process = []
    try:
        for i in range(args['worker']['num_gather']):
            conn = connect_socket_connection(args['worker']['remote_host'], 9998)
            p = mp.Process(target=gather_loop, args=(args, conn, i))
            p.start()
            conn.close()
            process.append(p)
        while True:
            time.sleep(100)
    finally:
        for p in process:
            p.terminate()
