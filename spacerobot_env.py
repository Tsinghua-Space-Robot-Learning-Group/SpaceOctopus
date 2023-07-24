import numpy as np

from gym.envs.robotics import utils
import robot_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SpacerobotEnv(robot_env.RobotEnv):
    """Superclass for all SpaceRobot environments.
    """

    def __init__(
        self, model_path, n_substeps,
        distance_threshold, initial_qpos, reward_type,
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            ? gripper_extra_height (float): additional height above the table when positioning the gripper【定位gripper】
            ? block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            ? target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            ? target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions采样初始目标位置的均匀分布范围
            ? target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
#        self.gripper_extra_height = gripper_extra_height
#        self.block_gripper = block_gripper
#        self.has_object = has_object
#        self.target_in_the_air = target_in_the_air
#        self.target_offset = target_offset
        self.n_substeps = n_substeps
#        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super(SpacerobotEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps,# n_actions=4,
            initial_qpos=initial_qpos)    
    
    
    # GoalEnv methods
    # ----------------------------


    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        #print('action',action)
        assert action.shape == (6,) # 可改
        action = action.copy() # ensure that we don't change the action outside of this scope
        self.sim.data.ctrl[:] = action* 0.2
        for _ in range(self.n_substeps):
            self.sim.step()
        
    def _get_obs(self):

        # positions
        grip_pos = self.sim.data.get_body_xpos('tip_frame')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp('tip_frame') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # position = self.sim.data.qpos.flat.copy()
        # velocity = self.sim.data.qvel.flat.copy()
        
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-1:]
        gripper_vel = robot_qvel[-1:] * dt  # change to a scalar if the gripper is made symmetric
        
        achieved_goal = grip_pos.copy()
        # 观测量加入了goal
        obs = np.concatenate([
            self.sim.data.qpos[7:13].copy(), self.sim.data.qvel[6:12].copy(),
            grip_pos, grip_velp, self.goal.copy()
        ])
        # obs = np.concatenate([
        #     grip_pos,  grip_velp, self.goal.copy()
        # ])
        
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
    

    def _viewer_setup(self):
#        body_id = self.sim.model.body_name2id('forearm_link')
        body_id = self.sim.model.body_name2id('wrist_3_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.
    

    def _reset_sim(self): 
        self.sim.set_state(self.initial_state) # self.initial_state在robot_env中定义;环境重置之后，机械臂的位置回到初始自然state
        self.sim.forward()        
        return True

    def _sample_goal(self):
        # 鉴于机械臂的初始位置是0,需要按照初始情况分别设置target的x,y,z坐标
        goal = self.sim.data.get_body_xpos('tip_frame').copy()
        d = goal_distance(self.sim.data.get_body_xpos('tip_frame').copy(),goal)

        # TODO：你这个约束条件太多了 而且范围有点小 之后改成下面那种范围的
        # while d < 0.43 or d > 0.55 : # be sure the initpos > 0.45
        #     goal[0] = self.sim.data.get_body_xpos('tip_frame')[0] + self.np_random.uniform(-0.34, -0.27) # 目标为移动到随机位置
        #     goal[1] = self.sim.data.get_body_xpos('tip_frame')[1] + self.np_random.uniform(-0.20, 0.25)
        #     goal[2] = self.sim.data.get_body_xpos('tip_frame')[2] + self.np_random.uniform(0.30, 0.36)
        #
        #     d = goal_distance(self.sim.data.get_body_xpos('tip_frame').copy(),goal)
        #     #print('AD',d)

        goal[0] = self.initial_gripper_xpos[0] + np.random.uniform(-0.4,0)  # self.np_random.uniform(-0.45, 0) # 目标为移动到随机位置
        goal[1] = self.initial_gripper_xpos[1] + np.random.uniform(-0.3, 0.3)  # self.np_random.uniform(-0.3, 0.3)
        goal[2] = self.initial_gripper_xpos[2] + np.random.uniform(0, 0.3)  # self.np_random.uniform(0.1, 0.3)


        # 显示target的位置
        site_id = self.sim.model.site_name2id('target0') # 设置target的位置
        self.sim.model.site_pos[site_id] = goal        
        self.sim.forward()

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
        #return d

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items(): # 机械臂的初始joint状态设置
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_body_xpos('tip_frame').copy()
    

    def render(self, mode='human', width=500, height=500):
        return super(SpacerobotEnv, self).render(mode, width, height)