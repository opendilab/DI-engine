# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# evaluation of policies or planning algorithms

import random
import time
import yaml
import multiprocessing as mp

import numpy as np

from .model import load_model
from .environment import prepare_env, make_env
from .connection import send_recv, accept_socket_connections, connect_socket_connection
#from .agents.SACNN120.wrapper import WrappedAgent as SACNN120
#from .agents.Hybrid373.wrapper import WrappedAgent as Hybrid373
#from .agents.Hybrid684.wrapper import WrappedAgent as Hybrid684
#from .agents.Hybrid700.wrapper import WrappedAgent as Hybrid700
#from .agents.Hybrid741.wrapper import WrappedAgent as Hybrid741
#from .agents.Hybrid829.wrapper import WrappedAgent as Hybrid829
#from .agents.Hybrid855.wrapper import WrappedAgent as Hybrid855
#from .agents.Hybrid978.wrapper import WrappedAgent as Hybrid978
#from .agents.HybridAllowShot893.wrapper import WrappedAgent as HybridAllowShot893
#from .agents.HybridAllowShot1017.wrapper import WrappedAgent as HybridAllowShot1017
#from .agents.Hybrid610Builtin.wrapper import WrappedAgent as Hybrid610Builtin
#from .agents.HybridAllowShot1175.wrapper import WrappedAgent as HybridAllowShot1175
#from .agents.StickyUPGO991.wrapper import WrappedAgent as StickyUPGO991
#from .agents.StickyUPGO1550.wrapper import WrappedAgent as StickyUPGO1550
#from .agents.ScoreReward1428.wrapper import WrappedAgent as ScoreReward1428
#from .agents.NewBig163.wrapper import WrappedAgent as NewBig163


io_match_port = 9876


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)


class BuiltinAgent(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action(player)

class RuleBasedAgentA(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_A(player)

class RuleBasedAgentB(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_B(player)

class RuleBasedAgentC(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_C(player)

class RuleBasedAgentD(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_D(player)

class RuleBasedAgentE(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_E(player)

class RuleBasedAgentF(RandomAgent):
    def action(self, env, player, show=False):
        return env.rule_based_action_F(player)

class IdleAgent(RandomAgent):
    def action(self, env, player, show=False):
        return 14

class RightAgent(RandomAgent):
    def action(self, env, player, show=False):
        return 5


def softmax(p, actions):
    ep = np.exp(p)
    p = ep / ep.sum()
    mask = np.zeros_like(p)
    mask[actions] = 1
    p = (p + 1e-16) * mask
    p /= p.sum()
    return p


def view(env, player=None):
    if hasattr(env, 'view'):
        env.view(player=player)
    else:
        print(env)


def view_transition(env):
    if hasattr(env, 'view_transition'):
        env.view_transition()
    else:
        pass


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        print('v = %f' % v)
        print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, planner):
        # planner might be a neural nets, or some game tree search
        self.planner = planner
        self.hidden = None

    def reset(self, env, show=False):
        self.hidden = self.planner.init_hidden()

    def action(self, env, player, show=False):
        p, v, _, self.hidden = self.planner.inference(env.observation(player), self.hidden)
        actions = env.legal_actions(player)
        if show:
            view(env, player=player)
            print_outputs(env, softmax(p, actions), v)
        ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
        return ap_list[0][0]


class SoftAgent(Agent):
    def action(self, env, player, show=False):
        p, v, _, self.hidden = self.planner.inference(env.observation(player), self.hidden)
        actions = env.legal_actions(player)
        prob = softmax(p, actions)
        if show:
            view(env, player=player)
            print_outputs(env, prob, v)
        return random.choices(np.arange(len(p)), weights=prob)[0]


class IOAgentClient:
    def __init__(self, agent, env, conn, show=False):
        self.conn = conn
        self.agent = agent
        self.env = env
        self.show = show

    def run(self):
        while True:
            command, args = self.conn.recv()
            if command == 'quit':
                break
            elif command == 'outcome':
                print('outcome = %f' % args[0])
            elif hasattr(self.agent, command):
                ret = getattr(self.agent, command)(self.env, *args, show=self.show)
                if command == 'action':
                    ret = self.env.action2str(ret)
            else:
                ret = getattr(self.env, command)(*args)
                if command == 'play_info':
                    if self.show:
                        view_transition(self.env)
            self.conn.send(ret)


class IOAgent:
    def __init__(self, conn):
        self.conn = conn

    def reset(self, data):
        send_recv(self.conn, ('reset_info', [data]))
        return send_recv(self.conn, ('reset', []))

    def chance(self, data):
        return send_recv(self.conn, ('chance_info', [data]))

    def play(self, data):
        return send_recv(self.conn, ('play_info', [data]))

    def outcome(self, outcome):
        return send_recv(self.conn, ('outcome', [outcome]))

    def action(self, player):
        return send_recv(self.conn, ('action', [player]))


def exec_match(env, agents, show=False, game_args={}):
    ''' match with shared game environment '''
    if env.reset(game_args):
        return None
    for agent in agents.values():
        agent.reset(env, show=show)
    while not env.terminal():
        if env.chance():
            return None
        if env.terminal():
            break
        actions = [0] * len(agents)
        for p, agent in agents.items():
            actions[p] = agent.action(env, p, show=show)
        if env.plays(actions):
            return None
        if show:
            view_transition(env)
    outcome = env.outcome()
    if show:
        print('final outcome = %s' % outcome)
    return [np.sign(o) for o in outcome]


def exec_io_match(env, io_agents, show=False, game_args={}):
    ''' match with divided game environment '''
    if env.reset(game_args):
        return None
    info = env.diff_info()
    for agent in io_agents.values():
        agent.reset(info)
    while not env.terminal():
        if env.chance():
            return None
        if env.terminal():
            break
        actions = [0] * len(io_agents)
        for p, agent in io_agents.items():
            actions[p] = env.str2action(agent.action(p))
        if env.plays(actions):
            return None
        info = env.diff_info()
        for agent in io_agents.values():
            agent.play(info)
    outcome = env.outcome()
    for p, agent in io_agents.items():
        agent.outcome(outcome[p])
    return [np.sign(o) for o in outcome]


class Evaluator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.default_agent = BuiltinAgent()

    def execute(self, models, args):
        agents = {}
        opponent_name = ''
        for p, model in models.items():
            if model is None:
                agent_map = {
                    (IdleAgent, 'donothing'): 1,
                    (RuleBasedAgentB, 'rulebaseB'): 2,
                    (RuleBasedAgentC, 'rulebaseC'): 4,
                    #(RuleBasedAgentD, 'rulebaseD'): 2,
                    (RuleBasedAgentE, 'rulebaseE'): 1,
                    (BuiltinAgent, 'builtinAI'): 7,
                    #(SACNN120, 'SACNN180'): 2,
                    #(Hybrid373, 'Hybrid373'): 2,
                    #(Hybrid684, 'Hybrid684'): 1,
                    #(Hybrid700, 'Hybrid700'): 1,
                    #(Hybrid741, 'Hybrid741'): 2,
                    #(Hybrid829, 'Hybrid829'): 2,
                    #(Hybrid855, 'Hybrid855'): 2,
                    #(Hybrid978, 'Hybrid978'): 2,
                    #(HybridAllowShot893, 'HybridAllowShot893'): 2,
                    #(HybridAllowShot1017, 'HybridAllowShot1017'): 2,
                    #(Hybrid610Builtin, 'Hy610Built'): 2,
                    #(HybridAllowShot1175, 'HybridAllowShot1175'): 3,
                    #(StickyUPGO991, 'StickyUPGO991'): 1,
                    #(StickyUPGO1550, 'StickyUPGO1550'): 3,
                    #(ScoreReward1428, 'ScoreReward1428'): 4,
                    #(NewBig163, 'NewBig163'): 2,
                }
                def normalize(w):
                    s = sum(w)
                    return [p / s for p in w]
                opponent = random.choices(list(agent_map.keys()), k=1, weights=normalize(list(agent_map.values())))[0]
                agents[p] = opponent[0]()
                opponent_name = opponent[1]
            else:
                agents[p] = Agent(model)
        outcome = exec_match(self.env, agents, game_args=args)
        if outcome is None:
            print('None episode in evaluation!')
            return None
        else:
            outcome = outcome[args['player'][0]]
        return opponent_name, outcome


def wp_func(results):
    games = sum([v for k, v in results.items() if k is not None])
    win = sum([(k + 1) / 2 * v for k, v in results.items() if k is not None])
    if games == 0:
        return 0.0
    return win / games


def eval_process_mp_child(agents, env_args, index, in_queue, out_queue, seed, show=False):
    random.seed(seed + index)
    env = make_env({**env_args, 'id': index})
    while True:
        args = in_queue.get()
        if args is None:
            break
        g, agent_ids, pat_idx, game_args = args
        print('*** Game %d ***' % g)
        agent_map = {env.players()[p]: agents[ai] for p, ai in enumerate(agent_ids)}
        if isinstance(list(agent_map.values())[0], IOAgent):
            outcome = exec_io_match(env, agent_map, show=show, game_args=game_args)
        else:
            outcome = exec_match(env, agent_map, show=show, game_args=game_args)
        out_queue.put((pat_idx, agent_ids, outcome))
    out_queue.put(None)


def evaluate_mp(env_args, agents, args_patterns, num_process, num_games, seed):
    env = make_env(env_args)
    in_queue, out_queue = mp.Queue(), mp.Queue()
    args_cnt = 0
    total_results, result_map = [{} for _ in agents], [{} for _ in agents]
    print('total games = %d' % (len(args_patterns) * num_games))
    time.sleep(0.1)
    for pat_idx, args in args_patterns.items():
        for i in range(num_games):
            if len(agents) == 2:
                # When playing two player game,
                # the number of games with first or second player is equalized.
                first_agent = 0 if i < (num_games // 2) else 1
                tmp_pat_idx, agent_ids = (pat_idx + '-F', [0, 1]) if first_agent == 0 else (pat_idx + '-S', [1, 0])
            else:
                tmp_pat_idx, agent_ids = pat_idx, random.sample(list(range(len(agents))), len(agents))
            in_queue.put((args_cnt, agent_ids, tmp_pat_idx, args))
            for p in range(len(agents)):
                result_map[p][tmp_pat_idx] = {}
            args_cnt += 1

    io_mode = agents[0] is None
    if io_mode:  # network battle mode
        agents = io_match_acception(num_process, env_args, len(agents), io_match_port)
    else:
        agents = [agents] * num_process

    for i in range(num_process):
        in_queue.put(None)
        args = agents[i], env_args, i, in_queue, out_queue, seed
        if num_process > 1:
            mp.Process(target=eval_process_mp_child, args=args).start()
            if io_mode:
                for agent in agents[i]:
                    agent.conn.close()
        else:
            eval_process_mp_child(*args, show=True)

    finished_cnt = 0
    while finished_cnt < num_process:
        ret = out_queue.get()
        if ret is None:
            finished_cnt += 1
            continue
        pat_idx, agent_ids, outcome = ret
        if outcome is not None:
            for idx, p in enumerate(env.players()):
                agent_id = agent_ids[idx]
                oc = outcome[p]
                result_map[agent_id][pat_idx][oc] = result_map[agent_id][pat_idx].get(oc, 0) + 1
                total_results[agent_id][oc] = total_results[agent_id].get(oc, 0) + 1

    for p, r_map in enumerate(result_map):
        print('---agent %d---' % p)
        for pat_idx, results in r_map.items():
            print(pat_idx, {k: results[k] for k in sorted(results.keys(), reverse=True)}, wp_func(results))
        print('total', {k: total_results[p][k] for k in sorted(total_results[p].keys(), reverse=True)}, wp_func(total_results[p]))
    return total_results


def io_match_acception(n, env_args, num_agents, port):
    waiting_conns = []
    accepted_conns = []

    for conn in accept_socket_connections(port):
        if len(accepted_conns) >= n * num_agents:
            break
        waiting_conns.append(conn)

        if len(waiting_conns) == num_agents:
            conn = waiting_conns[0]
            accepted_conns.append(conn)
            waiting_conns = waiting_conns[1:]
            conn.send(env_args)  # send accpept with environment arguments

    agents_list = [
        [IOAgent(accepted_conns[i * num_agents + j]) for j in range(num_agents)]
        for i in range(n)
    ]

    return agents_list


def get_model(env, model_path):
    import torch
    from .model import DuelingNet as Model
    model = env.net()(env) if hasattr(env, 'net') else Model(env)
    model = load_model(model, model_path)
    model.eval()
    return model


def client_mp_child(env_args, model_path, conn, show):
    env = make_env(env_args)
    model = get_model(env, model_path)
    IOAgentClient(Agent(model), env, conn, show).run()


def eval_main(args, argv):
    env_args = args['env_args']
    prepare_env(env_args)
    env = make_env(env_args)

    model_path = argv[0]
    agent1 = Agent(get_model(env, model_path))

    opponent_agents = {
        'builtinAI': BuiltinAgent(),
        #'rulebaseD': RuleBasedAgentD(),
        'rulebaseC': RuleBasedAgentC(),
        'rulebaseB': RuleBasedAgentB()
    }

    num_games = int(argv[1]) if len(argv) >= 2 else 100
    num_process = int(argv[2]) if len(argv) >= 3 else 8

    print('%d process, %d games' % (num_process, num_games))

    seed = random.randrange(1e8)
    print('seed = %d' % seed)

    total_results_map = {}
    for name, opponent in opponent_agents.items():
        print('vs', name)
        agents = [agent1, opponent]
        total_results_map[name] = evaluate_mp(env_args, agents, {'detault': {}}, num_process, num_games, seed)

    for name, total_results in total_results_map.items():
        results = total_results[0]
        print(name, {k: results[k] for k in sorted(results.keys(), reverse=True)}, wp_func(results))


def eval_server_main(args, argv):
    env_args = args['env_args']
    prepare_env(env_args)
    env = make_env(env_args)

    num_games = int(argv[0]) if len(argv) >= 1 else 100
    num_process = int(argv[1]) if len(argv) >= 2 else 8

    print('%d process, %d games' % (num_process, num_games))

    seed = random.randrange(1e8)
    print('seed = %d' % seed)

    print('io-match server mode')
    evaluate_mp(env_args, [None] * len(env.players()), {'default': {}}, num_process, num_games, seed)


def eval_client_main(args, argv):
    print('io-match client mode')
    prepared = False
    while True:
        try:
            conn = connect_socket_connection(args['eval_args']['remote_host'], io_match_port)
            env_args = conn.recv()
        except EOFError:
            break

        if not prepared:
            prepare_env(env_args)
            prepared = True

        model_path = argv[0]
        mp.Process(target=client_mp_child, args=(env_args, model_path, conn, False)).start()
        conn.close()
