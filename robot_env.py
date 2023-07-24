import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

import mujoco_py


DEFAULT_SIZE = 500

class RobotEnv(gym.GoalEnv):
    # n_actions是actuator的数量（区别RL算法中的n_actions）
    # n_substeps ?
    def __init__(self, model_path, initial_qpos, n_substeps):

        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        
        self._env_setup(initial_qpos=initial_qpos) # 设置Robot的初始姿态及target（各joint的初始角度）
        self.initial_state = copy.deepcopy(self.sim.get_state())

        
        self.goal = self._sample_goal() # 根据环境中是否有待抓取的目标确定任务目标
        obs = self._get_obs()
        
        self._set_action_space() # action_space若要严格参照model中对各个joint的控制信号的限制
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

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

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action): # mujoco中没有step函数,是自己定义的
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action) # _set_action(action)函数【在SpaceRobot环境中具体定义】,类比mujoco中的do_simulation()函数
        self.sim.step() # set action之后step【mujoco-py中用于模拟的API】        
        self._step_callback() # 回调函数不一定要
        obs = self._get_obs()

        success_info = self._is_success(obs['achieved_goal'], self.goal)
        if success_info == 0.:
            done = False
        else:
            done = True
        info = {
            'is_success': success_info,
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)        
        return obs, reward, done, info

    
    def reset(self): # 重置环境，可在每一个episode结束后调用
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues【数值问题】(e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition【初始条件】 (e.g. an object is within the hand).
        # In this case, we just keep randomizing【保持随机】 until we eventually achieve a valid initial
        # configuration.【有效的初始配置】
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim() # _reset_sim()【在低一级环境中定义】

        self.goal = self._sample_goal() # every reset get another goal
        obs = self._get_obs()
        return obs
    

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        #self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()
    

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)            
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.【重置模拟，并指示reset是否成功】
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """        
        self.sim.set_state(self.initial_state) # reset时将状态设置为规定的initial_state
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
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