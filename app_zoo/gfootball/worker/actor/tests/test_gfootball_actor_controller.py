import os
import time
from threading import Thread

import pytest

from nervex.worker.actor import create_actor, register_actor
from app_zoo.gfootball.worker.actor.gfootball_actor_controller import GfootballActor


class FakeGfootballActor(GfootballActor):

    def _setup_agents(self):
        super()._setup_agents()

        def no_op_fn(*args, **kwargs):
            pass

        for a in self._agents.values():
            a.state_dict = lambda: {'model': 'placeholder'}
            a.load_state_dict = no_op_fn


@pytest.mark.envtest
class TestASActorFakeEnv:

    def test_naive(self, setup_config, setup_coordinator, setup_manager):
        os.popen("rm -rf job_*")
        os.popen("rm -rf actor-log")
        time.sleep(1)
        register_actor("fake_gfootball", FakeGfootballActor)
        setup_config.actor.actor_type = "fake_gfootball"
        controller = create_actor(setup_config)

        def run():
            controller.run()

        run_thread = Thread(target=run, args=())
        run_thread.daemon = True
        run_thread.start()
        time.sleep(3)
        controller.close()
        time.sleep(3)
        print('end')
        os.popen("rm -rf job_*")
        os.popen("rm -rf actor-log")
