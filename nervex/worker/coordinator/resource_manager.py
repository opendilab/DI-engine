import random


class NaiveResourceManager(object):

    def __init__(self) -> None:
        self._worker_type = ['actor', 'learner']
        self._resource_info = {k: {} for k in self._worker_type}

    def assign_actor(self, actor_task: dict) -> dict:
        return {'actor_id': random.sample(self._resource_info['actor'].keys(), 1)[0]}

    def assign_learner(self, learner_task: dict) -> dict:
        return {'learner_id': random.sample(self._resource_info['learner'].keys(), 1)[0]}

    def update(self, name: str, worker_id: str, resource_info: dict) -> None:
        assert name in self._worker_type
        self._resource_info[name][worker_id] = resource_info
