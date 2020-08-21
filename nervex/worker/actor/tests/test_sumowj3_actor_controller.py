import pytest
from sensewow.worker.actor.alphastar_actor_controller import AlphaStarActor
from nervex.worker.actor.sumowj3_actor_controller import SumoWJ3Actor


class FakeSumoWJ3Actor(SumoWJ3Actor):
    def _setup_agents(self):
        super()._setup_agents()

        def no_op_fn(*args, **kwargs):
            pass

        for a in self._agents.values():
            a.state_dict = lambda: {'model': 'placeholder'}
            a.load_state_dict = no_op_fn
        print('setup agents over')


@pytest.mark.unittest
class TestASActorFakeEnv:
    def test_naive(self, setup_config, setup_coordinator, setup_manager):
        comm_cfg = setup_config.actor.communication
        controller = FakeSumoWJ3Actor(setup_config, comm_cfg=comm_cfg)
        controller.run()
