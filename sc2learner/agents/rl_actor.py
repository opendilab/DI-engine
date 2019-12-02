from queue import Queue
from threading import Thread
import zmq
import torch
from sc2learner.agents.ppo_policies_pytorch import MlpPolicy, LstmPolicy


class BaseActor(object):

    def __init__(self, env, model, unroll_length,
                 enable_push, queue_size,
                 learner_ip="localhost", port=None):
        assert(isinstance(learner_ip, str))  # TODO(nyz) support multi learner ip
        self.env = env
        self.model = model  # TODO(nyz) whether create model inside
        self.unroll_length = unroll_length

        self.zmq_context = zmq.Context()
        self.model_requestor = self.zmq_context.socket(zmq.REQ)
        self.model_requestor.connect("tcp://%s:%s" % (learner_ip, port['learner']))
        if enable_push:
            self.data_queue = Queue(queue_size)
            self.push_thread = Thread(target=self._push_data, args=(self.zmq_context,
                                      learner_ip, port['actor'], self.data_queue))
            self.push_thread.start()
        self.enable_push = enable_push

        self._init()

    def run(self):
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


class PpoActor(BaseActor):
    def __init__(self, *args, gamma=None, lam=None, **kwargs):
        super(PpoActor, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.lam = lam
        if isinstance(self.model, MlpPolicy):
            self.model_type = 'mlp'
        elif isinstance(self.model, LstmPolicy):
            self.model_type = 'lstm'
        else:
            raise ValueError

    # overwrite
    def _nstep_rollout(self):
        output_items = ['obs', 'action', 'value', 'neglogp', 'done', 'reward']
        if self.model.use_mask:
            output_items.append('mask')
        outputs = {k: [] for k in output_items}
        episode_infos = []
        for _ in range(self.unroll_length):
            inputs = self._pack_model_input()
            self._save_model_input(inputs, outputs)
            with torch.no_grad():
                model_output = self.model(inputs, mode='step')
            action = self._process_model_output(model_output, outputs)
            self.obs, reward, self.done, info = self.env.step(action)
            outputs['reward'].append(reward)
            self.cumulative_reward += reward
            if self.done:
                episode_infos.append(self.cumulative_reward)
                self._init()

        inputs = self._pack_model_input()
        with torch.no_grad():
            last_values = self.model(inputs, mode='value').squeeze(0)
        outputs['return'] = self._get_return(outputs, last_values)

        outputs['state'] = self.state
        outputs['episode_infos'] = episode_infos
        return outputs

    def _get_return(self, outputs, last_values):
        last_gae_lam = 0  # TODO clarify name
        returns = outputs['value'].copy()
        for i in reversed(range(self.unroll_length)):
            if i == self.unroll_length - 1:
                next_nontermial = 1.0 - self.done
                next_values = last_values
            else:
                next_nontermial = 1.0 - outputs['done'][i + 1]
                next_values = outputs['value'][i + 1]
            delta = (outputs['reward'][i] +
                     self.gamma * next_values * next_nontermial -
                     outputs['value'][i])
            last_gae_lam = (delta +
                            self.gamma * self.lam * next_nontermial * last_gae_lam)
            returns[i] += last_gae_lam
        return returns

    def _pack_model_input(self):
        inputs = {}
        if self.model.use_mask:
            obs, mask = self.obs
            inputs['mask'] = torch.FloatTensor(mask)
        else:
            obs = self.obs
        obs = torch.FloatTensor(obs)

        inputs['obs'] = obs.unsqueeze(0)
        done = torch.FloatTensor([self.done])
        inputs['done'] = done.unsqueeze(0)
        if self.model_type == 'lstm':
            inputs['state'] = self.state.unsqueeze(0)
        return inputs

    def _save_model_input(self, inputs, outputs):
        obs, done = inputs['obs'], inputs['done']
        outputs['obs'].append(obs.squeeze(0))
        outputs['done'].append(done.squeeze(0))
        if self.model.use_mask:
            mask = inputs['mask']
            outputs['mask'].append(mask)

    def _process_model_output(self, output, outputs):
        action, value, state, neglogp = output
        self.state = state
        action = action.squeeze(0)
        outputs['action'].append(action)
        outputs['value'].append(value.squeeze(0))
        outputs['neglogp'].append(neglogp.view(1))

        return action.numpy()

    # overwrite
    def _init(self):
        self.obs = self.env.reset()
        self.done = False
        self.state = self.model.initial_state
        self.cumulative_reward = 0
