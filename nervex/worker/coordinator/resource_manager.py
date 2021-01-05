import random


class NaiveResourceManager(object):

    def __init__(self) -> None:
        self._worker_type = ['actor', 'learner']
        self._resource_info = {k: {} for k in self._worker_type}

    def assign_actor(self, actor_task: dict) -> dict:
        available_actor = list(self._resource_info['actor'].keys())
        if len(available_actor) > 0:
            selected_actor = random.sample(available_actor, 1)[0]
            self._resource_info['actor'].pop(selected_actor)
            return {'actor_id': selected_actor}
        else:
            return None

    def assign_learner(self, learner_task: dict) -> dict:
        available_learner = list(self._resource_info['learner'].keys())
        if len(available_learner) > 0:
            selected_learner = random.sample(available_learner, 1)[0]
            self._resource_info['learner'].pop(selected_learner)
            return {'learner_id': selected_learner}
        else:
            return None

    def update(self, name: str, worker_id: str, resource_info: dict) -> None:
        assert name in self._worker_type
        self._resource_info[name][worker_id] = resource_info
