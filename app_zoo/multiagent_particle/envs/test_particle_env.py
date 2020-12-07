import pytest
import torch
from app_zoo.multiagent_particle.envs import ParicleEnv


@pytest.mark.unittest
class TestParticleEnv:

    def test_simple(self):
        self.env_test('simple')

    def test_simple_push(self):
        self.env_test('simple_push')

    def test_simple_tag(self):
        self.env_test('simple_tag')

    def test_simple_spread(self):
        self.env_test('simple_spread')

    def test_simple_adversary(self):
        self.env_test('simple_adversary')

    def test_simple_crypto(self):
        self.env_test('simple_crypto')

    def test_simple_reference(self):
        self.env_test('simple_reference', False)

    def test_simple_speaker_listener(self):
        self.env_test('simple_speaker_listener')

    def test_simple_world_comm(self):
        self.env_test('simple_world_comm', False)

    def env_test(self, name, doprint=False):
        env = ParicleEnv({"env_name": name})
        if doprint:
            print(env.info())
        obs = env.reset()

        for i in range(env.agent_num):
            assert obs[i].shape == env.info().obs_space['agent' + str(i)].shape

        #try run randomly for 100 step
        for _ in range(100):
            random_action = []
            for i in range(env.agent_num):
                act_sp = env.info().act_space.get('agent' + str(i))
                if act_sp:
                    act_val = act_sp.value
                    min_val, max_val = act_val['min'], act_val['max']
                    if act_sp.shape == (1, ):
                        random_action.append(torch.randint(min_val, max_val, act_sp.shape))
                    else:
                        random_action.append(
                            torch.stack([torch.randint(min_val[t], max_val[t], (1, )) for t in range(act_sp.shape[0])])
                        )
            timestep = env.step(random_action)
            if doprint:
                print(timestep)
            for i in range(env.agent_num):
                assert timestep.obs[i].size() == env.info().obs_space['agent' + str(i)].shape
                assert timestep.reward[i].size() == env.info().rew_space['agent' + str(i)].shape
            assert isinstance(timestep, tuple)
        env.close()
