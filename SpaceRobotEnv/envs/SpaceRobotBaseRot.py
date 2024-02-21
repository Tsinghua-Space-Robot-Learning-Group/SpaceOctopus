import os

import copy
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.robotics import utils
from gym.envs.robotics import rotations

import mujoco_py

PATH = os.getcwd()

MODEL_XML_PATH = os.path.join(PATH,'SpaceRobotEnv','assets', 'spacerobot', 'spacerobot_fourarm.xml')
DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_substeps):

        # load model and simulator
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)

        # render setting
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        # seed
        self.seed()

        # initalization
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()

        # set action_space and observation_space
        obs = self._get_obs()
        self._set_action_space()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _detecte_collision(self):
        self.collision = self.sim.data.ncon
        return self.collision

    def _sensor_torque(self):
        self.sensor_data = self.sim.data.sensordata
        return self.sensor_data

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert action.shape == (24,)
        old_action = self.sim.data.ctrl.copy() * (1 / 0.5)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)  # do one step simulation here
        self._step_callback()
        colli = self._detecte_collision()
        obs = self._get_obs()
        done = False
        info = { 
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "act": action,
            "old_act": old_action,
            }
        reward = self.compute_reward(
            # obs["achieved_goal"], self.goal.copy(), action, info
            obs["achieved_goal"], self.goal.copy(), action, old_action, colli, info
        )
        return obs, reward, done, info

    def reset(self):
        """Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.
        """
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.goal = self._sample_goal()
        obs = self._get_obs()

        # TODO: set the position of cube

        # body_id = self.sim.model.geom_name2id("cube")
        # self.sim.model.geom_pos[body_id] = np.array([0, 0, 6])
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)

        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
                self._viewer_setup()

            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
                self._viewer_setup()
                # self.viewer.cam.trackbodyid = 0
                # latest modification
                cam_pos = np.array([0, 0, 4, 4.2, -15, 160])
                for i in range(3):
                    self.viewer.cam.lookat[i] = cam_pos[i]
                self.viewer.cam.distance = cam_pos[3]
                self.viewer.cam.elevation = cam_pos[4]
                self.viewer.cam.azimuth = cam_pos[5]
                # self.viewer.cam.trackbodyid = -1

            self._viewers[mode] = self.viewer
        return self.viewer

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it is successful.
        If a reset is unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback【自定义回调】 that is called before rendering. Can be used
        to implement custom visualizations.【可实现自定义可视化】
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.【对模拟状态强制附加约束】
        """
        pass


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SpacerobotEnv(RobotEnv):
    """Superclass for all SpaceRobot environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        distance_threshold,
        initial_qpos,
        reward_type,
        pro_type, 
        c_coeff
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            pro_type ('MDP' or 'CMDP'):  the problem setting whether contains cost or not
            c_coeff: cost coefficient
        """
        self.n_substeps = n_substeps
        #        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.pro_type = pro_type
        self.reward_type = reward_type
        self.c_coeff = c_coeff

        super(SpacerobotEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos,
        )

    def compute_reward(self, achieved_goal, desired_goal, action, old_action, colli, info):
        d = goal_distance(achieved_goal, desired_goal)
        loss1 = np.linalg.norm(action - old_action) ** 2
        loss2 = np.linalg.norm(action) ** 2

        reward = {
            "sparse": -(d > self.distance_threshold).astype(np.float32),
            "dense": - (np.log10(d ** 2 + 1e-6) + 0.01 * loss1 + 0.05 * loss2 + 0.5 * colli),
        }
        # print("r0: ",reward["r0"],"r1: ",reward["r1"],"r2: ",reward["r2"],"r3: ",reward["r3"])
        # print("r0=", 0.001 * rd1 ** 2 , np.log10(rd1 ** 2 + 1e-6) , 0.01 * l0)

        return reward

    def _set_action(self, action):
        """
        output action (velocity)
        :param action: angle velocity of joints
        :return: angle velocity of joints
        """
        act = action.copy()  # ensure that we don't change the action outside of this scope
        self.sim.data.ctrl[:] = act * 0.5
        for _ in range(self.n_substeps):
            self.sim.step()

    def _get_obs(self):
        post_base_att = self.sim.data.get_body_xquat('chasersat')
        post_base_att = rotations.quat2euler(post_base_att)

        qpos_tem = self.sim.data.qpos[:31].copy()
        qvel_tem = self.sim.data.qvel[:30].copy()

        obs = np.concatenate(
            [
                qpos_tem,
                qvel_tem,
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob0 = np.concatenate(
            [
                qpos_tem[:10].copy(),
                qvel_tem[:9].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob1 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[10:13].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[9:12].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob2 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[13:16].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[12:15].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob3 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[16:19].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[15:18].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob4 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[19:22].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[18:21].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob5 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[22:25].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[21:24].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob6 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[25:28].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[24:27].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        ob7 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[28:31].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[27:30].copy(),
                post_base_att.copy(),
                self.goal.copy(),
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": post_base_att.copy(),
            "desired_goal": self.goal.copy(),
            "observation_0":ob0.copy(),
            "observation_1":ob1.copy(),
            "observation_2":ob2.copy(),
            "observation_3":ob3.copy(), 
            "observation_4":ob4.copy(), 
            "observation_5":ob5.copy(), 
            "observation_6":ob6.copy(), 
            "observation_7":ob7.copy()
        }

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('forearm_link')
        body_id = self.sim.model.body_name2id("wrist_3_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = np.array([0, 0, 0], dtype=np.float32)
        goal[0] = self.initial_base_att[0] + np.random.uniform(-0.40, 0.40)
        goal[1] = self.initial_base_att[1] + np.random.uniform(-0.40, 0.40)
        goal[2] = self.initial_base_att[2] + np.random.uniform(-0.40, 0.40)
        site_id = self.sim.model.site_name2id("targetbase")
        self.sim.model.site_pos[site_id] = np.array([0, 0, 4], dtype=np.float32)
        self.sim.model.site_quat[site_id] = rotations.euler2quat(goal.copy())
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
        # return d

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)

        # get the initial base attitude
        initial_base_att = self.sim.data.get_body_xquat("chasersat").copy()
        self.initial_base_att = rotations.quat2euler(initial_base_att)

        # get the initial base position
        self.initial_base_pos = self.sim.data.get_body_xpos("chasersat").copy()

    def render(self, mode="human", width=500, height=500):
        return super(SpacerobotEnv, self).render(mode, width, height)


class SpaceRobotBaseRot(SpacerobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type="sparse", pro_type = 'MDP'):
        initial_qpos = {
            "arm:shoulder_pan_joint": 1.57*2,
            "arm:shoulder_lift_joint": -1.57,
            "arm:elbow_joint": 0.0,
            "arm:wrist_1_joint": -1.57,
            "arm:wrist_2_joint": 0.0,
            "arm:wrist_3_joint": 1.57*2,
            "arm:shoulder_pan_joint1": 0.0,
            "arm:shoulder_lift_joint1": -1.57,
            "arm:elbow_joint1": 0.0,
            "arm:wrist_1_joint1": -1.57,
            "arm:wrist_2_joint1": 0.0,
            "arm:wrist_3_joint1": 0.0,
            "arm:shoulder_pan_joint2": 0.0,
            "arm:shoulder_lift_joint2": -1.57,
            "arm:elbow_joint2": 0.0,
            "arm:wrist_1_joint2": -1.57,
            "arm:wrist_2_joint2": 0.0,
            "arm:wrist_3_joint2": 0.0,
            "arm:shoulder_pan_joint3": 1.57*2,
            "arm:shoulder_lift_joint3": -1.57,
            "arm:elbow_joint3": 0.0,
            "arm:wrist_1_joint3": -1.57,
            "arm:wrist_2_joint3": 0.0,
            "arm:wrist_3_joint3": 1.57*2,
        }
        SpacerobotEnv.__init__(
            self,
            MODEL_XML_PATH,
            n_substeps=20,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            pro_type=pro_type, 
            c_coeff=0.1,
        )
        gym.utils.EzPickle.__init__(self)
