from collections import namedtuple, OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import copy
from pysc2.lib.static_data import BUILD_ORDER_REWARD_ACTIONS, UNIT_BUILD_ACTIONS, EFFECT_ACTIONS, RESEARCH_ACTIONS
from sc2learner.data.collate_fn import diff_shape_collate
from sc2learner.torch_utils import levenshtein_distance, hamming_distance, to_device


class RewardHelper:
    BattleValues = namedtuple('BattleValues', ['last_h', 'cur_h', 'last_a', 'cur_a'])

    def __init__(self, agent_num, pseudo_reward_type, pseudo_reward_prob):
        assert agent_num == 2
        self.agent_num = agent_num
        self.pseudo_reward_prob = pseudo_reward_prob
        self.pseudo_reward_type = pseudo_reward_type
        assert self.pseudo_reward_type in ['global', 'immediate']
        self.last_behaviour_z = None
        self.batch_size = self.agent_num
        self.device = torch.device('cpu')
        self.build_order_location_max_limit = 2
        self.build_order_location_rescale = 0.8

    def get_pseudo_rewards(
        self, rewards, action_types, episode_stats, loaded_eval_stats, game_loop, battle_values, return_list=False
    ):
        def check(t):
            return isinstance(t, list) and len(t) == self.agent_num

        assert check(rewards) and check(action_types) and check(episode_stats) and check(loaded_eval_stats)
        assert isinstance(battle_values, self.BattleValues)
        game_second = game_loop // 22
        # single player pseudo rewards
        if loaded_eval_stats[0].excess_max_game_loop(game_loop):  # differnet agents have the same game_loop
            rewards = self._get_zero_rewards(rewards)
        else:
            behaviour_zs = []
            human_target_zs = []
            for i in range(self.agent_num):
                if self.pseudo_reward_type == 'global':
                    behaviour_z = episode_stats[i].get_reward_z(use_max_bo_clip=True)
                    bo_length = len(behaviour_z['build_order']['type'])
                    human_target_z = loaded_eval_stats[i].get_reward_z_by_game_loop(
                        game_loop=None, build_order_length=bo_length
                    )
                elif self.pseudo_reward_type == 'immediate':
                    behaviour_z = episode_stats[i].get_reward_z(use_max_bo_clip=False)
                    human_target_z = loaded_eval_stats[i].get_reward_z_by_game_loop(game_loop=game_loop)
                behaviour_zs.append(behaviour_z)
                human_target_zs.append(human_target_z)
            game_seconds = [game_second] * self.agent_num
            behaviour_zs = diff_shape_collate(behaviour_zs)
            human_target_zs = diff_shape_collate(human_target_zs)
            masks = self._get_reward_masks(action_types, behaviour_zs, self.last_behaviour_z)

            rewards = self._compute_pseudo_rewards(behaviour_zs, human_target_zs, rewards, game_seconds, masks)
            self.last_behaviour_z = copy.deepcopy(behaviour_zs)

        # multi players pseudo rewards
        if self.agent_num == 2:
            rewards = self._compute_battle_reward(rewards, battle_values)
        if return_list:
            rewards = [{k: rewards[k][i].unsqueeze(0) for k in rewards.keys()} for i in range(self.agent_num)]
        return rewards

    def _get_zero_rewards(self, ori_rewards):
        rewards = {}
        rewards['winloss'] = torch.FloatTensor(ori_rewards)
        for k in ['build_order', 'built_unit', 'upgrade', 'effect']:
            rewards[k] = torch.zeros(self.batch_size)
        rewards = to_device(rewards, self.device)
        return rewards

    def _get_reward_masks(self, action_type, behaviour_z, last_behaviour_z):
        cum_stat_list = ['built_unit', 'effect', 'upgrade']
        action_map = {'built_unit': UNIT_BUILD_ACTIONS, 'effect': EFFECT_ACTIONS, 'upgrade': RESEARCH_ACTIONS}
        masks = {}
        mask_build_order = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            mask_build_order[i] = 1 if action_type[i] in BUILD_ORDER_REWARD_ACTIONS else 0
        masks['build_order'] = mask_build_order
        if last_behaviour_z is None:
            last_behaviour_z = {k: torch.zeros_like(v) for k, v in behaviour_z.items() if k in cum_stat_list}
            for k in cum_stat_list:
                mask = torch.zeros(self.batch_size)
                for i in range(self.batch_size):
                    if action_type[i] in action_map[k]:
                        mask[i] = 1
                masks[k] = mask
        else:
            for k in cum_stat_list:
                ne_num = behaviour_z[k].ne(last_behaviour_z[k]).sum(dim=1)
                masks[k] = torch.where(ne_num > 0, torch.ones(self.batch_size), torch.zeros(self.batch_size))
        masks = to_device(masks, self.device)
        return masks

    def _compute_pseudo_rewards(self, behaviour_z, human_target_z, rewards, game_seconds, masks):
        """
            Overview: compute pseudo rewards from human replay z
            Arguments:
                - behaviour_z (:obj:`dict`)
                - human_target_z (:obj:`dict`)
                - rewards (:obj:`torch.Tensor`)
                - game_seconds (:obj:`int`)
            Returns:
                - rewards (:obj:`dict`): a dict contains different type rewards
        """
        def loc_fn(p1, p2, max_limit=self.build_order_location_max_limit):
            p1 = p1.float().to(self.device)
            p2 = p2.float().to(self.device)
            dist = F.l1_loss(p1, p2, reduction='sum')
            dist = dist.clamp(0, max_limit)
            dist = dist / max_limit * self.build_order_location_rescale
            return dist.item()

        def get_time_factor(game_second):
            if game_second < 8 * 60:
                return 1.0
            elif game_second < 16 * 60:
                return 0.5
            elif game_second < 24 * 60:
                return 0.25
            else:
                return 0

        factors = torch.FloatTensor([get_time_factor(s) for s in game_seconds]).to(self.device)

        new_rewards = OrderedDict()
        new_rewards['winloss'] = torch.FloatTensor(rewards).to(self.device)
        # build_order
        p = np.random.uniform()
        build_order_reward = []
        for i in range(self.batch_size):
            # only proper action can activate
            mask = masks['build_order'][i]
            # only some prob can activate
            mask = mask if p < self.pseudo_reward_prob else 0
            # if current the length of the behaviour_build_order is longer than that of human_target_z, return zero
            if (len(behaviour_z['build_order']['type'][i]) > len(human_target_z['build_order']['type'][i])
                    and self.pseudo_reward_type == 'global'):
                build_order_reward.append(torch.FloatTensor([0]))
            else:
                build_order_reward.append(
                    -levenshtein_distance(
                        behaviour_z['build_order']['type'][i], human_target_z['build_order']['type'][i],
                        behaviour_z['build_order']['loc'][i], human_target_z['build_order']['loc'][i], loc_fn
                    ) * factors[i] * mask
                )
        new_rewards['build_order'] = torch.FloatTensor(build_order_reward).to(self.device)
        # built_unit, effect, upgrade
        # p is independent from all the pseudo reward and the same in a batch
        for k in ['built_unit', 'effect', 'upgrade']:
            mask = masks[k]
            p = np.random.uniform()
            mask_factor = 1 if p < self.pseudo_reward_prob else 0
            mask *= mask_factor
            new_rewards[k] = -hamming_distance(behaviour_z[k], human_target_z[k], factors) * mask
        for k in new_rewards.keys():
            new_rewards[k] = new_rewards[k].float()
        return new_rewards

    def _compute_battle_reward(self, rewards, battle_values):
        last_h, cur_h, last_a, cur_a = battle_values
        v = (cur_h - last_h) - (cur_a - last_a)
        v = torch.FloatTensor([v]).to(self.device)
        rewards['battle'] = torch.cat([v, -v])
        return rewards
