import gym
from gym import spaces
import numpy as np
# from envs.env_core import EnvCore
import SpaceRobotEnv
import gym

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class DualArmWithRot(object):
    """
    DualArm environment
    """

    def __init__(self,args):
        self.env = gym.make("SpaceRobotDualArmWithRot-v0")
        self.num_agents = args.num_agents
        self.share_reward = args.share_reward
        self.single_obs_dim = 31
        self.single_action_dim = 3
        self.share_observation_dim = self.single_obs_dim * self.num_agents

        # configure spaces
        bounds = self.env.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for i in range(self.num_agents):
            u_action_space = spaces.Box(
                low=low[self.single_action_dim*i:self.single_action_dim*(i+1)],
                high=high[self.single_action_dim*i:self.single_action_dim*(i+1)],
                dtype=np.float32,
            )
            self.action_space.append(u_action_space)
            # observation space
            u_observationn_space = spaces.Box(-np.inf, np.inf, shape=(self.single_obs_dim,),dtype=np.float32)
            self.observation_space.append(u_observationn_space)

        self.share_observation_space = [
            spaces.Box(-np.inf, np.inf, shape=(self.share_observation_dim,),dtype=np.float32) for _ in range(self.num_agents)
        ]

    def step(self, actions):
        
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        
        #action:a list, contains num_agent elements,each element is a (single_action_dim,)shape array. 
        a = np.stack(actions)
        assert a.shape == (self.num_agents,self.single_action_dim)
        act = a.reshape(12,)
        observation, reward, done, info = self.env.step(act)
        for i in range(self.num_agents):
            sub_agent_obs.append(observation["observation_"+str(i)])
            sub_agent_reward.append(reward["r"+str(i)])
            sub_agent_done.append(done)
            sub_agent_info.append(info)
        if self.share_reward:
            reward_sum = np.sum(sub_agent_reward)
            sub_agent_reward = [[reward_sum]] * self.num_agents
        return np.stack(sub_agent_obs), np.stack(sub_agent_reward), np.stack(sub_agent_done), sub_agent_info

    def reset(self):
        sub_agent_obs = []
        observation = self.env.reset()
        for i in range(self.num_agents):
            sub_agent_obs.append(observation["observation_"+str(i)])
        return np.stack(sub_agent_obs)

    def close(self):
        return self.env.close

    def render(self, mode="human"):
        return self.env.render(mode)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
