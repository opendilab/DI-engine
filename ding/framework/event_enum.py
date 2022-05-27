import re
from enum import Enum, unique


@unique
class EventEnum(Enum):
    # template event
    TEMPLATE_EVENT = "example_xxx_{actor_id}_{player_id}"

    # events emited by coordinators
    COORDINATOR_DISPATCH_ACTOR_JOB = "coordinator_dispatch_actor_job_{actor_id}"

    # events emited by learners
    LEARNER_SEND_MODEL = "on_learner_send_model"
    LEARNER_SEND_META = "on_learner_send_meta"

    # events emited by actors
    ACTOR_GREETING = "actor_greeting"
    ACTOR_SEND_DATA = "actor_send_meta"
    ACTOR_FINISH_JOB = "actor_finish_job"

    def get_event(self, *args, **kwargs):
        args_dict = {}
        if args:
            params = re.findall(r"\{(.*?)\}", self.value)
            args_dict.update(zip(params, args))
        args_dict.update(kwargs)
        return self.value.format(**args_dict)
