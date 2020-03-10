import torch
from sc2learner.torch_utils import build_checkpoint_helper
from .actor import BaseActor
from sc2learner.agent.model import PPOLSTM, PPOMLP
from sc2learner.envs.raw_env import SC2RawEnv
from sc2learner.envs.rewards.reward_wrappers import KillingRewardWrapper
from sc2learner.envs.actions.zerg_action_wrappers import ZergActionWrapper
from sc2learner.envs.observations.zerg_observation_wrappers \
    import ZergObservationWrapper


def create_env(cfg, difficulty, random_seed=None):
    env = SC2RawEnv(map_name='AbyssalReef',
                    step_mul=cfg.env.step_mul,
                    resolution=16,
                    agent_race='zerg',
                    bot_race='zerg',
                    difficulty=difficulty,
                    disable_fog=cfg.env.disable_fog,
                    tie_to_lose=False,
                    game_steps_per_episode=cfg.env.game_steps_per_episode,
                    random_seed=random_seed)
    if cfg.env.use_reward_shaping:
        env = KillingRewardWrapper(env)
    env = ZergActionWrapper(env,
                            game_version=cfg.env.game_version,
                            mask=cfg.env.use_action_mask,
                            use_all_combat_actions=cfg.env.use_all_combat_actions)
    env = ZergObservationWrapper(env,
                                 use_spatial_features=False,
                                 use_game_progress=(
                                     not cfg.model.policy == 'lstm'),
                                 action_seq_len=1 if cfg.model.policy == 'lstm' else 8,
                                 use_regions=cfg.env.use_region_features)
    env.difficulty = difficulty
    return env


class PpoActor(BaseActor):
    def __init__(self, *args, **kwargs):
        super(PpoActor, self).__init__(*args, **kwargs)
        self.gamma = self.cfg.train.discount_gamma
        self.lam = self.cfg.train.lambda_return
        self.model_type = self.cfg.model.policy

    # overwrite
    def _nstep_rollout(self):
        output_items = ['obs', 'action', 'value', 'neglogp', 'done', 'reward']
        if self.model.use_mask:
            output_items.append('mask')
        outputs = {k: [] for k in output_items}
        episode_infos = []
        outputs['state'] = self.state  # rollout begin state
        for _ in range(self.unroll_length):
            inputs = self._pack_model_input()
            self._save_model_input(inputs, outputs)
            with torch.no_grad():
                model_output = self.model(inputs, mode='step')
            action = self._process_model_output(model_output, outputs)
            self.obs, reward, self.done, info = self.env.step(action)
            outputs['reward'].append(reward)
            self.step += 1  # notice this is only the steps of action, not actual game step
            self.cumulative_reward += reward
            if self.done:
                episode_infos.append({'game_result': self.cumulative_reward,
                                      'difficulty': self.env.difficulty,
                                      'game_length': self.step})
                break
        inputs = self._pack_model_input()
        with torch.no_grad():
            last_values = self.model(inputs, mode='value')['value'].squeeze(0)
        outputs['return'] = self._get_return(outputs, last_values)
        outputs['episode_infos'] = episode_infos
        return outputs

    def _get_return(self, outputs, last_values):
        last_gae_lam = 0  # TODO clarify name
        returns = [t.clone() for t in outputs['value']]  # IMPORTANT
        this_rollout_len = len(outputs['obs'])
        for i in reversed(range(this_rollout_len)):
            if i == this_rollout_len - 1:
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
            inputs['mask'] = torch.FloatTensor(mask).unsqueeze(0)
        else:
            obs = self.obs[0]
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
            mask = inputs['mask'].squeeze(0)
            outputs['mask'].append(mask)

    def _process_model_output(self, output, outputs):
        action, value, state, neglogp = (
            output['action'], output['value'], output['state'], output['neglogp'])
        self.state = state
        action = action.squeeze(0)
        outputs['action'].append(action)
        outputs['value'].append(value.squeeze(0))
        outputs['neglogp'].append(neglogp)

        return action.numpy()

    # overwrite
    def _init(self):
        # Setting up env and model
        self._create_env()
        if self.model is None:
            self._create_model()
        self.model.set_seed(self.seed)
        self.obs = self.env.reset()
        self.done = False
        self.job_cancelled = False
        self.state = self.model.initial_state
        self.cumulative_reward = 0
        self.step = 0

    # overwrite
    def _create_env(self):
        if self.env is not None:
            self.env.close()
        job = self._request_job()
        self.job_id = job['job_id']
        self.start_rollout_at = job['start_rollout_at']
        self.seed = job['game_vs_bot']['seed']
        self.env = create_env(self.cfg,
                              job['game_vs_bot']['difficulty'],
                              self.seed)

    # overwrite
    def _create_model(self):
        policy_func = {'mlp': PPOMLP,
                       'lstm': PPOLSTM}
        self.model = policy_func[self.cfg.model.policy](
            ob_space=self.env.observation_space,
            ac_space=self.env.action_space,
            seed=self.seed
        )
