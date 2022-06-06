from enum import Enum, unique


@unique
class EventEnum(str, Enum):
    # events emited by coordinators
    COORDINATOR_DISPATCH_ACTOR_JOB = "on_coordinator_dispatch_actor_job_{actor_id}"

    # events emited by learners
    LEARNER_SEND_MODEL = "on_learner_send_model"
    LEARNER_SEND_META = "on_learner_send_meta"

    # events emited by actors
    ACTOR_GREETING = "on_actor_greeting"
    ACTOR_SEND_DATA = "on_actor_send_meta_player_{player}"
    ACTOR_FINISH_JOB = "on_actor_finish_job"
