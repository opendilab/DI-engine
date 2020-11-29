from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import q_1step_td_data_continuous, q_1step_td_error_continuous
from nervex.worker.agent import BaseAgent


class PendulumDdpgGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.ddpg.discount_factor
        self._actor_update_freq = cfg.ddpg.get('actor_update_freq', 2)
        self._forward_cnt = 0
        self._noise_type = cfg.get('noise_type', 'gauss')
        self._noise_kwargs = cfg.get('noise_kwargs', {
            'mu': 0.0,
            'sigma': 1.0,
            'range': 2.0
        })
        self._action_range = cfg.get('action_range', {
            'min': -2.0,
            'max': 2.0
        })

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done').float()
        weights = data.get('weights', None)

        # q
        q_value = agent.forward(data, param={'mode': 'compute_q'})['q_value']
        # target_q
        next_data = {'obs': next_obs}
        next_action = agent.forward(
            next_data,
            param={'mode': 'compute_action'},
            noise_type=self._noise_type,
            noise_kwargs=self._noise_kwargs,
            action_range=self._action_range
        )['action']
        next_data['action'] = next_action
        if agent.is_double:
            target_q_value = agent.target_forward(next_data, param={'mode': 'compute_q'})['q_value']
        else:
            target_q_value = agent.forward(next_data, param={'mode': 'compute_q'})['q_value']
        # critic_loss: q 1step td error
        td_data = q_1step_td_data_continuous(q_value, target_q_value, action, reward, done)
        critic_loss = q_1step_td_error_continuous(td_data, self._gamma, weights)
        actor_update = self._forward_cnt % self._actor_update_freq == 0
        # actor_loss: q grad ascent
        if actor_update:
            actor_loss = -agent.forward(data, param={'mode': 'optimize_actor'})['q_value'].mean()

        if agent.is_double:
            agent.target_update(agent.state_dict()['model'])
        self._forward_cnt += 1

        loss_dict = {}
        if actor_update:
            loss_dict['actor_loss'] = actor_loss
        for i, loss in enumerate(critic_loss):
            loss_dict['critic{}_loss'.format(str(i + 1))] = loss
        total_loss = sum([_ for _ in loss_dict.values()])
        loss_dict['total_loss'] = total_loss
        return loss_dict

    def __repr__(self) -> str:
        return "PendulumDdpgGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('actor_loss')
        recorder.register_var('critic1_loss')
        recorder.register_var('critic2_loss')
        recorder.register_var('total_loss')
        tb_logger.register_var('actor_loss')
        tb_logger.register_var('critic1_loss')
        tb_logger.register_var('critic2_loss')
        tb_logger.register_var('total_loss')
