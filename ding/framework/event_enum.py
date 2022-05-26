from enum import Enum, unique


@unique
class EventEnum(Enum):
    COORDINATOR_DISPATCH_ACTOR_JOB = 1000

    LEARNER_SEND_MODEL = 2000
    LEARNER_SEND_META = 2001

    ACTOR_GREETING = 3000
    ACTOR_SEND_DATA = 3001
    ACTOR_FINISH_JOB = 3002

    def __call__(self, node_id: int = None):
        if not node_id:
            return "event_{}".format(self.value)
        return "event_{}_{}".format(self.value, node_id)