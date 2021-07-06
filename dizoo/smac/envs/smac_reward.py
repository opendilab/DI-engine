from collections import namedtuple
import numpy as np

ORIGINAL_AGENT = "me"
OPPONENT_AGENT = "opponent"


class SMACReward:
    info_template = namedtuple('EnvElementInfo', ['shape', 'value', 'to_agent_processor', 'from_agent_processor'])

    def __init__(
        self,
        n_agents,
        n_enemies,
        two_player,
        reward_type,
        max_reward,
        reward_scale=True,
        reduce_agent=True,
        reward_only_positive=True
    ):
        self.reward_only_positive = reward_only_positive
        self.reward_scale = reward_scale
        self.max_reward = max_reward
        self.reward_death_value = 10
        self.reward_win = 200
        self.reward_defeat = 0
        self.reward_negative_scale = 0.5
        self.reward_scale_rate = 20
        self.reduce_agent = reduce_agent
        self.reward_type = reward_type
        assert self.reward_type in ['sparse', 'original', 'new']
        self.n_agents = n_agents
        self.n_enemies = n_enemies

        self.death_tracker_ally = np.zeros(n_agents)
        self.death_tracker_enemy = np.zeros(n_enemies)

        self.two_player = two_player

    def reset(self, max_reward):
        self.max_reward = max_reward
        if self.reward_type == 'original':
            self.info().value['max'] = self.max_reward / self.reward_scale_rate
        self.death_tracker_ally.fill(0)
        self.death_tracker_enemy.fill(0)

    def get_reward(self, engine, action, game_end_code, win_counted, defeat_counted):
        reward = {
            ORIGINAL_AGENT: np.asarray(self.reward_battle_split(engine, action, is_opponent=False)),
            OPPONENT_AGENT: np.asarray(self.reward_battle_split(engine, action, is_opponent=True))
        }
        for k in reward:
            if reward[k].shape == ():
                reward[k] = np.expand_dims(reward[k], 0)

        if game_end_code is not None:
            # Battle is over
            if game_end_code == 1 and not win_counted:
                if self.reward_type != "sparse":
                    reward[ORIGINAL_AGENT] += self.reward_win
                    reward[OPPONENT_AGENT] += self.reward_defeat
                else:
                    reward[ORIGINAL_AGENT] += 1
                    reward[OPPONENT_AGENT] += -1
            elif game_end_code == -1 and not defeat_counted:
                if self.reward_type != "sparse":
                    reward[ORIGINAL_AGENT] += self.reward_defeat
                    reward[OPPONENT_AGENT] += self.reward_win
                else:
                    reward[ORIGINAL_AGENT] += -1
                    reward[OPPONENT_AGENT] += 1
            # Note: if draw happen, the game_end_code may still be None.

        if self.reward_scale:
            # rescale to 0~1
            min_val, max_val = self.info().value['min'], self.info().value['max']
            reward[ORIGINAL_AGENT] = (reward[ORIGINAL_AGENT] - min_val) / (max_val - min_val)
            reward[OPPONENT_AGENT] = (reward[OPPONENT_AGENT] - min_val) / (max_val - min_val)

        return reward

    def reward_battle_split(self, engine, action, is_opponent=False):
        """Reward function when self.reward_type != 'sparse'.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """

        num_agents = engine.n_agents if not is_opponent else engine.n_enemies
        num_enmies = engine.n_agents if is_opponent else engine.n_enemies

        if self.reward_type == 'sparse':
            if self.reduce_agent:
                return 0.
            else:
                return np.zeros(num_agents)

        # if self.reward_type != 'original':
        assert self.reward_type == 'original', 'reward_type={} is not supported!'.format(self.reward_type)
        delta_deaths = np.zeros([num_agents])
        reward = np.zeros([num_agents])
        delta_ally = np.zeros([num_agents])
        delta_enemy = np.zeros([num_enmies])
        delta_death_enemy = np.zeros([num_enmies])

        neg_scale = self.reward_negative_scale

        # update deaths
        if is_opponent:
            iterator = engine.enemies.items()
            previous_units = engine.previous_enemy_units
            death_tracker = self.death_tracker_enemy
        else:
            iterator = engine.agents.items()
            previous_units = engine.previous_ally_units
            death_tracker = self.death_tracker_ally

        num_players = 2 if self.two_player else 1
        for al_id, al_unit in iterator:
            if death_tracker[al_id] < num_players:
                # did not die so far
                prev_health = (previous_units[al_id].health + previous_units[al_id].shield)
                if al_unit.health == 0:
                    # just died
                    death_tracker[al_id] += 1
                    delta_deaths[al_id] -= self.reward_death_value * neg_scale
                    delta_ally[al_id] += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally[al_id] += neg_scale * (prev_health - al_unit.health - al_unit.shield)

        # Calculate the damage to opponent.
        if is_opponent:
            iterator = engine.agents.items()
            previous_units = engine.previous_ally_units
            death_tracker = self.death_tracker_ally
        else:
            iterator = engine.enemies.items()
            previous_units = engine.previous_enemy_units
            death_tracker = self.death_tracker_enemy

        for e_id, e_unit in iterator:
            if death_tracker[e_id] < num_players:
                prev_health = (previous_units[e_id].health + previous_units[e_id].shield)
                if e_unit.health == 0:
                    death_tracker[e_id] += 1
                    delta_death_enemy[e_id] += self.reward_death_value
                    delta_enemy[e_id] += prev_health
                else:
                    delta_enemy[e_id] += prev_health - e_unit.health - e_unit.shield
                # if e_unit.health == 0:
                #     death_tracker[e_id] += 1
                #     delta_death_enemy[e_id] += self.reward_death_value
                #     normed_delta_health = prev_health / (e_unit.health_max + e_unit.shield_max)
                #     delta_enemy[e_id] += normed_delta_health * self.reward_death_value
                # else:
                #     normed_delta_health = (prev_health - e_unit.health -
                #                            e_unit.shield) / (e_unit.health_max + e_unit.shield_max)
                #     delta_enemy[e_id] += normed_delta_health * self.reward_death_value

        # if self.reward_type == 'original':
        #     if self.reduce_agent:
        #         total_reward = sum(delta_deaths) + sum(delta_death_enemy) + sum(delta_enemy)
        #         return total_reward
        #     else:
        #         total_reward = sum(delta_deaths) + sum(delta_death_enemy) + sum(delta_enemy) / num_agents
        #         return np.ones(num_agents) * total_reward

        # Attacking reward
        # if isinstance(action, dict):
        #     my_action = action["me"] if not is_opponent else action["opponent"]
        # else:
        #     my_action = action
        # for my_id, my_action in enumerate(my_action):
        #     if my_action > 5:
        #         reward[my_id] += 2

        if self.reward_only_positive:
            # reward = abs((delta_deaths + delta_death_enemy + delta_enemy).sum())
            reward = abs(delta_deaths.sum() + delta_death_enemy.sum() + delta_enemy.sum())
        else:
            reward = delta_deaths.sum() + delta_death_enemy.sum() + delta_enemy.sum() - delta_ally.sum()

        return reward

    def info(self):
        if self.reward_type == 'sparse':
            value = {'min': -1, 'max': 1}
        elif self.reward_type == 'original':
            value = {'min': 0, 'max': self.max_reward / self.reward_scale_rate}
            # value = {'min': 0, 'max': 75.5}
            # value = {'min': 0, 'max': self.max_reward / 75.5}
        #     # TODO(nyz) health + shield range
        #     if self.reduce_agent:
        #         value = {'min': 0, 'max': (self.reward_win + self.reward_death_value * self.n_enemies +1230)/20}
        #     else:
        #         value = {'min': 0, 'max': self.reward_win + self.reward_death_value * self.n_enemies / self.n_agents}
        # elif self.reward_type == 'new':
        #     if self.reduce_agent:
        #         value = {'min': 0, 'max': self.reward_win + 2 + self.reward_death_value * self.n_enemies}
        #     else:
        #         value = {
        #             'min': 0,
        #             'max': self.reward_win + 2 + self.reward_death_value * self.n_enemies / self.n_agents
        #         }
        shape = (1, ) if self.reduce_agent else (self.n_agents, )
        return SMACReward.info_template(shape, value, None, None)
