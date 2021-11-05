from typing import Any


class Context(dict):
    """
    Overview:
        Context is an object that pass contextual data between middlewares, whose life cycle is only one step.
        It is a dict that reflect itself, so you can set any properties as you wish.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.total_step = 0  # Total steps
        self.step = 0  # Step in current episode
        self.episode = 0  # Total episodes
        self.state = None  # State received from env
        self.next_state = None  # Next state from env
        self.action = None  # Action
        self.reward = None  # Reward
        self.done = False  # Whether current step is the last step of current episode
        self.policy_output = None  # Policy output

        # Reserved properties
        self._backward_stack = []
        self._finish = False
        self._kept = {"_finish": True}

    def set_default(self, key: str, value: Any, keep=False) -> None:
        """
        Overview:
            Set default value of a property, if you want to keep this property to the next iteration,
            set keep=True.
        Arguments:
            - key (:obj:`str`): The key.
            - value (:obj:`Any`): The value.
            - keep (:obj:`bool`): Whether to keep this attribute until the next iteration.
        """
        if key not in self:
            self[key] = value
        if keep:
            self._kept[key] = True
