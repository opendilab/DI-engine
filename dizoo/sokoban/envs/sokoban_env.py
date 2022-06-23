import gym
import random
import marshal
import copy
import numpy as np
from collections import namedtuple
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from ding.utils import ENV_REGISTRY
from ding.envs import ObsPlusPrevActRewWrapper


@ENV_REGISTRY.register('sokoban')
class SokobanEnv(BaseEnv):

    timestep = namedtuple('Sokoban', ['obs', 'reward', 'done', 'info'])

    ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
    }

    CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
    }

    explored_states = set()
    num_boxes = 0
    best_room_score = -1
    best_room = None
    best_box_mapping = None

    
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        self.render_mode = cfg.rendor_mode
        self.observation_mode = cfg.observation_mode

        self.penalty_for_step =cfg.penalty_for_step
        self.reward_box_on_target = cfg.reward_box_on_target
        self.penalty_box_off_target = cfg.penalty_box_off_target
        self.reward_finished = cfg.reward_finished

        self.dim_room = cfg.dim_room
        self.max_steps = cfg.max_steps
        if cfg.num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (self.dim_room[0] + self.dim_room[1]))
        else:
            self.num_gen_steps = cfg.num_gen_steps

        self._action_space = Discrete(len(SokobanEnv.ACTION_LOOKUP))
        screen_height, screen_width = (self._cfg.dim_room[0] * 16, self._cfg.dim_room[1] * 16)
        self._observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

        self.reset()

    def reset(self):
        self._env = gym.make(self._env_id)
        if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
            self._env = ObsPlusPrevActRewWrapper(self._env)
        

        self.room_fixed, self.room_state, self.box_mapping = self._generate_room(
            dim=self.dim_room,
            num_steps=self.num_gen_steps,
            num_boxes=self.num_boxes,
            second_player=False)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(self.render_mode)
        return starting_observation

    def step(self, action):
        assert action in SokobanEnv.ACTION_LOOKUP
        assert self.observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=self.observation_mode)

        info = {
            "action.name": self.ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return SokobanEnv.timestep(
            obs=observation,
            reward = self.reward_last,
            done = done,
            info = info
        )

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = self.CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = self.CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _generate_room(self, dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False):
        room_state = np.zeros(shape=dim)
        room_structure = np.zeros(shape=dim)

        # Some times rooms with a score == 0 are the only possibility.
        # In these case, we try another model.
        for t in range(tries):
            room = self._room_topology_generation(dim, p_change_directions, num_steps)
            room = self._place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

            # Room fixed represents all not movable parts of the room
            room_structure = np.copy(room)
            room_structure[room_structure == 5] = 1

            # Room structure represents the current state of the room including movable parts
            room_state = room.copy()
            room_state[room_state == 2] = 4

            room_state, score, box_mapping = self._reverse_playing(room_state, room_structure)
            room_state[room_state == 3] = 4

            if score > 0:
                break

        if score == 0:
            raise RuntimeWarning('Generated Model with score == 0')

        return room_structure, room_state, box_mapping

    def _room_topology_generation(self, dim=(10, 10), p_change_directions=0.35, num_steps=15):
        """
        Generate a room topology, which consits of empty floors and walls.

        :param dim:
        :param p_change_directions:
        :param num_steps:
        :return:
        """
        dim_x, dim_y = dim

        # The ones in the mask represent all fields which will be set to floors
        # during the random walk. The centered one will be placed over the current
        # position of the walk.
        masks = [
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0]
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ],
            [
                [0, 0, 0],
                [1, 1, 0],
                [1, 1, 0]
            ],
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 0]
            ]
        ]

        # Possible directions during the walk
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        direction = random.sample(directions, 1)[0]

        # Starting position of random walk
        position = np.array([
            random.randint(1, dim_x - 1),
            random.randint(1, dim_y - 1)]
        )

        level = np.zeros(dim, dtype=int)

        for s in range(num_steps):

            # Change direction randomly
            if random.random() < p_change_directions:
                direction = random.sample(directions, 1)[0]

            # Update position
            position = position + direction
            position[0] = max(min(position[0], dim_x - 2), 1)
            position[1] = max(min(position[1], dim_y - 2), 1)

            # Apply mask
            mask = random.sample(masks, 1)[0]
            mask_start = position - 1
            level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

        level[level > 0] = 1
        level[:, [0, dim_y - 1]] = 0
        level[[0, dim_x - 1], :] = 0

        return level

    def _place_boxes_and_player(self, room, num_boxes, second_player):
        """
        Places the player and the boxes into the floors in a room.

        :param room:
        :param num_boxes:
        :return:
        """
        # Get all available positions
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]
        num_players = 2 if second_player else 1

        if num_possible_positions <= num_boxes + num_players:
            raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
                num_possible_positions,
                num_players,
                num_boxes)
            )

        # Place player(s)
        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

        if second_player:
            ind = np.random.randint(num_possible_positions)
            player_position = possible_positions[0][ind], possible_positions[1][ind]
            room[player_position] = 5

        # Place boxes
        for n in range(num_boxes):
            possible_positions = np.where(room == 1)
            num_possible_positions = possible_positions[0].shape[0]

            ind = np.random.randint(num_possible_positions)
            box_position = possible_positions[0][ind], possible_positions[1][ind]
            room[box_position] = 2

        return room

    def _reverse_playing(self, room_state, room_structure, search_depth=100):
        """
        This function plays Sokoban reverse in a way, such that the player can
        move and pull boxes.
        It ensures a solvable level with all boxes not being placed on a box target.
        :param room_state:
        :param room_structure:
        :param search_depth:
        :return: 2d array
        """
        # Box_Mapping is used to calculate the box displacement for every box
        box_mapping = {}
        box_locations = np.where(room_structure == 2)
        num_boxes = len(box_locations[0])
        for l in range(num_boxes):
            box = (box_locations[0][l], box_locations[1][l])
            box_mapping[box] = box

        # explored_states globally stores the best room state and score found during search
        self.explored_states = set()
        self.best_room_score = -1
        self.best_box_mapping = box_mapping
        self._depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300)

        return self.best_room, self.best_room_score, self.best_box_mapping

    def _depth_first_search(self,room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300):
        """
        Searches through all possible states of the room.
        This is a recursive function, which stops if the tll is reduced to 0 or
        over 1.000.000 states have been explored.
        :param room_state:
        :param room_structure:
        :param box_mapping:
        :param box_swaps:
        :param last_pull:
        :param ttl:
        :return:
        """
        ttl -= 1
        if ttl <= 0 or len(self.explored_states) >= 300000:
            return

        state_tohash = marshal.dumps(room_state)

        # Only search this state, if it not yet has been explored
        if not (state_tohash in self.explored_states):

            # Add current state and its score to explored states
            room_score = box_swaps * self._box_displacement_score(box_mapping)
            if np.where(room_state == 2)[0].shape[0] != self.num_boxes:
                room_score = 0

            if room_score > best_room_score:
                best_room = room_state
                best_room_score = room_score
                best_box_mapping = box_mapping

            self.explored_states.add(state_tohash)

            for action in self.ACTION_LOOKUP.keys():
                # The state and box mapping  need to be copied to ensure
                # every action start from a similar state.
                room_state_next = room_state.copy()
                box_mapping_next = box_mapping.copy()

                room_state_next, box_mapping_next, last_pull_next = \
                    self._reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

                box_swaps_next = box_swaps
                if last_pull_next != last_pull:
                    box_swaps_next += 1

                self._depth_first_search(room_state_next, room_structure,
                                box_mapping_next, box_swaps_next,
                                last_pull, ttl)

    def _reverse_move(self,room_state, room_structure, box_mapping, last_pull, action):
        """
        Perform reverse action. Where all actions in the range [0, 3] correspond to
        push actions and the ones greater 3 are simmple move actions.
        :param room_state:
        :param room_structure:
        :param box_mapping:
        :param last_pull:
        :param action:
        :return:
        """
        player_position = np.where(room_state == 5)
        player_position = np.array([player_position[0][0], player_position[1][0]])

        change = self.CHANGE_COORDINATES[action % 4]
        next_position = player_position + change

        # Check if next position is an empty floor or an empty box target
        if room_state[next_position[0], next_position[1]] in [1, 2]:

            # Move player, independent of pull or move action.
            room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
            room_state[next_position[0], next_position[1]] = 5

            # In addition try to pull a box if the action is a pull action
            if action < 4:
                possible_box_location = change[0] * -1, change[1] * -1
                possible_box_location += player_position

                if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                    # Perform pull of the adjacent box
                    room_state[player_position[0], player_position[1]] = 3
                    room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                        possible_box_location[0], possible_box_location[1]]

                    # Update the box mapping
                    for k in box_mapping.keys():
                        if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                            box_mapping[k] = (player_position[0], player_position[1])
                            last_pull = k

        return room_state, box_mapping, last_pull

    def _box_displacement_score(self, box_mapping):
        """
        Calculates the sum of all Manhattan distances, between the boxes
        and their origin box targets.
        :param box_mapping:
        :return:
        """
        score = 0
        
        for box_target in box_mapping.keys():
            box_location = np.array(box_mapping[box_target])
            box_target = np.array(box_target)
            dist = np.sum(np.abs(box_location - box_target))
            score += dist

        return score

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

