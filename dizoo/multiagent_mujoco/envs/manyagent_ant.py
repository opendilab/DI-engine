import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from jinja2 import Template
import os

class ManyAgentAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        agent_conf = kwargs.get("agent_conf")
        n_agents = int(agent_conf.split("x")[0])
        n_segs_per_agents = int(agent_conf.split("x")[1])
        n_segs = n_agents * n_segs_per_agents

        # Check whether asset file exists already, otherwise create it
        asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                                                  'manyagent_ant_{}_agents_each_{}_segments.auto.xml'.format(n_agents,
                                                                                                                 n_segs_per_agents))
        #if not os.path.exists(asset_path):
        print("Auto-Generating Manyagent Ant asset with {} segments at {}.".format(n_segs, asset_path))
        self._generate_asset(n_segs=n_segs, asset_path=asset_path)

        #asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',git p
        #                          'manyagent_swimmer.xml')

        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)
        utils.EzPickle.__init__(self)

    def _generate_asset(self, n_segs, asset_path):
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets',
                                                  'manyagent_ant.xml.template')
        with open(template_path, "r") as f:
            t = Template(f.read())
        body_str_template = """
        <body name="torso_{:d}" pos="-1 0 0">
           <!--<joint axis="0 1 0" name="nnn_{:d}" pos="0.0 0.0 0.0" range="-1 1" type="hinge"/>-->
            <geom density="100" fromto="1 0 0 0 0 0" size="0.1" type="capsule"/>
            <body name="front_right_leg_{:d}" pos="0 0 0">
              <geom fromto="0.0 0.0 0.0 0.0 0.2 0.0" name="aux1_geom_{:d}" size="0.08" type="capsule"/>
              <body name="aux_2_{:d}" pos="0.0 0.2 0">
                <joint axis="0 0 1" name="hip1_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom_{:d}" size="0.08" type="capsule"/>
                <body pos="-0.2 0.2 0">
                  <joint axis="1 1 0" name="ankle1_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom_{:d}" size="0.08" type="capsule"/>
                </body>
              </body>
            </body>
            <body name="back_leg_{:d}" pos="0 0 0">
              <geom fromto="0.0 0.0 0.0 0.0 -0.2 0.0" name="aux2_geom_{:d}" size="0.08" type="capsule"/>
              <body name="aux2_{:d}" pos="0.0 -0.2 0">
                <joint axis="0 0 1" name="hip2_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom_{:d}" size="0.08" type="capsule"/>
                <body pos="-0.2 -0.2 0">
                  <joint axis="-1 1 0" name="ankle2_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom_{:d}" size="0.08" type="capsule"/>
                </body>
              </body>
            </body>
        """

        body_close_str_template ="</body>\n"
        actuator_str_template = """\t     <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip1_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle1_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip2_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle2_{:d}" gear="150"/>\n"""

        body_str = ""
        for i in range(1,n_segs):
            body_str += body_str_template.format(*([i]*16))
        body_str += body_close_str_template*(n_segs-1)

        actuator_str = ""
        for i in range(n_segs):
            actuator_str += actuator_str_template.format(*([i]*8))

        rt = t.render(body=body_str, actuators=actuator_str)
        with open(asset_path, "w") as f:
            f.write(rt)
        pass

    def step(self, a):
        xposbefore = self.get_body_com("torso_0")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso_0")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5