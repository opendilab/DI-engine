import pytest
import torch
from app_zoo.multiagent_particle.envs import ParticleEnv

use_discrete = [True, False]


@pytest.mark.unittest
@pytest.mark.parametrize('discrete_action', use_discrete)
class TestParticleEnv:

    def test_simple(self, discrete_action):
        self.env_test('simple', discrete_action)

    def test_simple_push(self, discrete_action):
        self.env_test('simple_push', discrete_action)

    def test_simple_tag(self, discrete_action):
        self.env_test('simple_tag', discrete_action)

    def test_simple_spread(self, discrete_action):
        self.env_test('simple_spread', discrete_action)

    def test_simple_adversary(self, discrete_action):
        self.env_test('simple_adversary', discrete_action)

    def test_simple_crypto(self, discrete_action):
        self.env_test('simple_crypto', discrete_action)

    def test_simple_reference(self, discrete_action):
        self.env_test('simple_reference', discrete_action, False)

    def test_simple_speaker_listener(self, discrete_action):
        self.env_test('simple_speaker_listener', discrete_action)

    def test_simple_world_comm(self, discrete_action):
        self.env_test('simple_world_comm', discrete_action, False)

    def env_test(self, name, discrete_action, doprint=False):
        env = ParticleEnv({"env_name": name, "discrete_action": discrete_action})
        if doprint:
            print(env.info())
        obs = env.reset()

        for i in range(env.agent_num):
            assert obs[i].shape == env.info().obs_space['agent' + str(i)].shape

        #try run randomly for 100 step
        for _ in range(100):
            random_action = []
            # print(env.info().act_space)
            for i in range(env.agent_num):
                act_sp = env.info().act_space.get('agent' + str(i))
                if act_sp:
                    act_val = act_sp.value
                    min_val, max_val = act_val['min'], act_val['max']
                    if act_sp.shape == (1, ):
                        if discrete_action:
                            random_action.append(torch.randint(min_val, max_val, act_sp.shape))
                        else:
                            random_action.append(torch.rand(max_val + 1 - min_val, ))
                    else:
                        # print(act_sp.shape)
                        if discrete_action:
                            random_action.append(
                                torch.cat(
                                    [torch.randint(min_val[t], max_val[t], (1, )) for t in range(act_sp.shape[0])]
                                )
                                # [torch.randint(min_val[t], max_val[t], (1, )) for t in range(act_sp.shape[0])]
                            )
                        else:
                            # print("i = ", i)
                            # print('randon_action = ', random_action)
                            # print([torch.rand(max_val[t]+1 - min_val[t], ) for t in range(act_sp.shape[0])])
                            random_action.append(
                                # torch.stack([torch.rand(max_val[t]+1 - min_val[t], ) for t in range(act_sp.shape[0])])
                                [torch.rand(max_val[t] + 1 - min_val[t], ) for t in range(act_sp.shape[0])]
                            )
            # print('randon_action = ', random_action)
            timestep = env.step(random_action)
            if doprint:
                print(timestep)
            for i in range(env.agent_num):
                assert timestep.obs[i].size() == env.info().obs_space['agent' + str(i)].shape
                assert timestep.reward[i].size() == env.info().rew_space['agent' + str(i)].shape
            assert isinstance(timestep, tuple)
        env.close()
