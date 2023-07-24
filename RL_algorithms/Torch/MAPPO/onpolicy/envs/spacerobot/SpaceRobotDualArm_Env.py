import gym
from gym import spaces
import numpy as np
# from envs.env_core import EnvCore
import SpaceRobotEnv
import gym

class DualArm(object):
    """
    DualArm environment
    """

    def __init__(self,args):
        self.env = gym.make("SpaceRobotDualArm-v0")
        self.num_agent = args.num_agents
        self.rewards_type = "dense"
        self.single_obs_dim = 28
        self.single_action_dim = 3
        self.share_observation_dim = 112

        # configure spaces
        bounds = self.env.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for i in range(self.num_agent):
            # physical action space
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
            spaces.Box(-np.inf, np.inf, shape=(self.share_observation_dim,),dtype=np.float32) for _ in range(self.num_agent)
        ]

    def ob_decomp(self,ob):
        assert (ob.shape == (55,))
        ob_ind = []
        ob_ind.append(np.concatenate([ob[0:10],ob[19:28],ob[37:40],ob[43:46],ob[49:52]]))
        ob_ind.append(np.concatenate([ob[0:7],ob[10:13],ob[19:25],ob[28:31],ob[37:40],ob[43:46],ob[49:52]]))
        ob_ind.append(np.concatenate([ob[0:7],ob[13:16],ob[19:25],ob[31:34],ob[40:43],ob[46:49],ob[52:55]]))
        ob_ind.append(np.concatenate([ob[0:7],ob[16:19],ob[19:25],ob[34:37],ob[40:43],ob[46:49],ob[52:55]]))
        return ob_ind

    def step(self, actions):
        
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        
        a = np.stack(actions)
        assert (a.shape == (self.num_agent,self.single_action_dim)),"check shape of actions, it has to be 4x3!"
        actions = a.reshape(12,)
        observation, reward, done, info = self.env.step(actions)
        ob = observation["observation"]
        ob_ind = self.ob_decomp(ob)
        for i in range(self.num_agent):
            sub_agent_obs.append(ob_ind[i])
            sub_agent_reward.append(reward[self.rewards_type])
            sub_agent_done.append(done)
            sub_agent_info.append(info)
        return np.stack(sub_agent_obs), np.stack(sub_agent_reward), np.stack(sub_agent_done), sub_agent_info

    def reset(self):
        ob = self.env.reset()["observation"]
        agent_obs = self.ob_decomp(ob)
        return np.stack(agent_obs)

    def close(self):
        return self.env.close

    def render(self, mode="rgb_array"):
        return self.env.render

    def seed(self, seed):
        return self.env.seed
