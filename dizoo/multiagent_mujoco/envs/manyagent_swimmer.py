import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from jinja2 import Template

class ManyAgentSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        agent_conf = kwargs.get("agent_conf")
        n_agents = int(agent_conf.split("x")[0])
        n_segs_per_agents = int(agent_conf.split("x")[1])
        n_segs = n_agents * n_segs_per_agents

        # Check whether asset file exists already, otherwise create it
        asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                                                  'manyagent_swimmer_{}_agents_each_{}_segments.auto.xml'.format(n_agents,
                                                                                                                 n_segs_per_agents))
        # if not os.path.exists(asset_path):
        print("Auto-Generating Manyagent Swimmer asset with {} segments at {}.".format(n_segs, asset_path))
        self._generate_asset(n_segs=n_segs, asset_path=asset_path)

        #asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',git p
        #                          'manyagent_swimmer.xml')

        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

    def _generate_asset(self, n_segs, asset_path):
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                                                  'manyagent_swimmer.xml.template')
        with open(template_path, "r") as f:
            t = Template(f.read())
        body_str_template = """
        <body name="mid{:d}" pos="-1 0 0">
          <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
          <joint axis="0 0 {:d}" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
        """

        body_end_str_template = """
        <body name="back" pos="-1 0 0">
            <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
            <joint axis="0 0 1" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
          </body>
        """

        body_close_str_template ="</body>\n"
        actuator_str_template = """\t <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="rot{:d}"/>\n"""

        body_str = ""
        for i in range(1,n_segs-1):
            body_str += body_str_template.format(i, (-1)**(i+1), i)
        body_str += body_end_str_template.format(n_segs-1)
        body_str += body_close_str_template*(n_segs-2)

        actuator_str = ""
        for i in range(n_segs):
            actuator_str += actuator_str_template.format(i)

        rt = t.render(body=body_str, actuators=actuator_str)
        with open(asset_path, "w") as f:
            f.write(rt)
        pass

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
