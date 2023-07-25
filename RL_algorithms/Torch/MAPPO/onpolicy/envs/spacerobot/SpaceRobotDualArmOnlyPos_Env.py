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
        self.single_obs_dim = 28
        self.single_action_dim = 3
        self.share_observation_dim = self.single_obs_dim * self.num_agents

        # configure spaces
        bounds = self.env.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        u_action_space_1 = spaces.Box(
            low=low[0:3],
            high=high[0:3],
            dtype=np.float32,
        )
        # u_action_space_2 = spaces.Box(
        #     low=low[3:6],
        #     high=high[3:6],
        #     dtype=np.float32,
        # )
        self.action_space.append(u_action_space_1)
        # self.action_space.append(u_action_space_2)

        for i in range(self.num_agents):
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
        a0 = actions[0]
        # print("action: ",a0)
        # a0 = actions
        a1 = np.zeros(3)
        # a1 = actions[1]
        a2 = np.zeros(3)
        # a2 = actions[1]
        a3 = np.zeros(3)
        a = [a0, a1, a2, a3]
        # print(a)
        a = np.stack(a)
        # print(a.shape)
        actions = a.reshape(12,)
        observation, reward, done, info = self.env.step(actions)
        # a = observation["achieved_goal"]
        # d = observation["desired_goal"]
        # rd1 = goal_distance(a[:3], d[:3])
        # rr1 = 0.1 * goal_distance(a[3:6], d[3:6])
        # print("achieved goal:",a,"\ndesired goal:",d)
        # print(observation["observation_0"][25:28])
        # print("r0:",- (0.001 * rd1 ** 2 + np.log10(rd1 ** 2 + 1e-6)))
        # print("r1:", - (0.001 * rr1 ** 2 + np.log10(rr1 ** 2 + 1e-6)))
        for i in range(self.num_agents):
            sub_agent_obs.append(observation["observation_"+str(i)])
            sub_agent_reward.append(reward["r"+str(i)])
            sub_agent_done.append(done)
            sub_agent_info.append(info)
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

    def seed(self, seed):
        return self.env.seed
