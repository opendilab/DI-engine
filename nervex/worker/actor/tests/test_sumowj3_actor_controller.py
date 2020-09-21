import pytest
import os
import threading
from threading import Thread
import time
from nervex.worker.actor.sumowj3_actor_controller import SumoWJ3Actor


class FakeSumoWJ3Actor(SumoWJ3Actor):
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
        comm_cfg = setup_config.actor.communication
        controller = FakeSumoWJ3Actor(setup_config, comm_cfg=comm_cfg)

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
