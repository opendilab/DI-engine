from itertools import product

import numpy as np


class BaseAgent:

    def __init__(self, break_value: float, delta_t: float):
        self.x = None
        self.y = None
        self.phi = None  # angle of the velocity vector
        self.theta = None  # direction of the agent
        self.speed = None
        self.delta_t = delta_t
        self.break_value = break_value

    def accelerate(self, value: float) -> None:
        raise NotImplementedError

    def break_(self) -> None:
        raise NotImplementedError

    def turn(self, value: float) -> None:
        raise NotImplementedError

    def reset(self, x: float, y: float, direction: float) -> None:
        self.x = x
        self.y = y
        self.speed = 0
        self.theta = direction

    def _step(self) -> None:
        angle = self.theta if self.phi is None else self.phi
        self.x += self.delta_t * self.speed * np.cos(angle)
        self.y += self.delta_t * self.speed * np.sin(angle)


class MovingAgent(BaseAgent):

    def __init__(self, break_value: float, delta_t: float):
        super(MovingAgent, self).__init__(break_value, delta_t)

    def accelerate(self, value: float) -> None:
        self.speed += value
        self._step()

    def break_(self) -> None:
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value
        self._step()

    def turn(self, value: float) -> None:
        self.theta = (self.theta + value) % (2 * np.pi)
        self._step()


class SlidingAgent(BaseAgent):

    def __init__(self, break_value: float, delta_t: float):
        super(SlidingAgent, self).__init__(break_value, delta_t)
        self.phi = 0

    def accelerate(self, value: float) -> None:
        # Adding two polar vectors: https://math.stackexchange.com/a/1365938/849658
        # phi_1, r_1 = self.theta, value  # the direction of the agent and the magnitude induced by the action
        # phi_2, r_2 = self.phi, self.speed  # the direction of the velocity vector and its magnitude
        speed = np.sqrt(value ** 2 + self.speed ** 2 + 2 * value * self.speed * np.cos(self.phi - self.theta))
        angle = self.theta + np.arctan2(
            self.speed * np.sin(self.phi - self.theta), value + self.speed * np.cos(self.phi - self.theta)
        )
        self.speed = speed
        self.phi = angle
        self._step()

    def break_(self) -> None:
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value
        self.phi = self.theta if self.speed == 0 else self.phi  # not sure it is needed
        self._step()

    def turn(self, value: float) -> None:
        self.theta = (self.theta + value) % (2 * np.pi)
        self._step()


class HardMoveAgent(BaseAgent):

    def __init__(self, break_value: float, delta_t: float, num_actuators: int = 4):
        super(HardMoveAgent, self).__init__(break_value, delta_t)
        self.phi = 0
        self.num_actuators = num_actuators
        # NOTE: meta_to_mask
        self.K = 2 ** self.num_actuators
        self.meta_to_mask = list(product(*[list(range(2)) for _ in range(self.num_actuators)]))

    def accelerate(self, value: float) -> None:
        pass

    def break_(self) -> None:
        pass

    def turn(self, value: float) -> None:
        pass

    def move(self, move_direction_meta: int, move_distances: list) -> None:
        move_directions_mask = self.meta_to_mask[int(move_direction_meta)]
        self.move_vector = np.array(
            [
                move_directions_mask[i] * move_distances[i] *
                np.array([np.cos(i * 2 * np.pi / self.num_actuators),
                          np.sin(i * 2 * np.pi / self.num_actuators)]) for i in range(len(move_distances))
            ]
        ).sum(0)
        self._step()
        self.theta = np.arctan(self.y / self.x)  # direction of the agent, in radian

    def _step(self) -> None:
        self.x = self.x + self.move_vector[0]
        self.y = self.y + self.move_vector[1]
