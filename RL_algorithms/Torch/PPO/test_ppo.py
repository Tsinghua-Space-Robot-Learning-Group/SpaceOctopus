import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
import numpy as np
import math
import os
import sys
import imageio

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)
import SpaceRobotEnv
from PPO_test import ActorCritic

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.temp = np.array([3.27954375e-01, 3.18709347e-01, 6.62526221e-01, 3.20982933e-01, \
 6.70563944e-01 ,3.31605755e+00, 3.76576439e-01, 4.06194717e-01, \
 4.74374438e-01, 2.91659507e-01, 3.84093806e-01, 5.06396706e-01, \
 1.23748465e-01, 1.75499651e-01, 1.16444766e-01, 2.68001657e-03, \
 3.92001154e-03, 5.44218938e-03, 1.14968292e-01, 1.73315928e-01, \
 8.66867990e-02])

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.temp + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

reward_list = []
    
env = gym.make("SpaceRobotState-v0")
num_inputs = env.observation_space["observation"].shape[0]
num_actions = env.action_space.shape[0]
running_state = ZFilter((num_inputs,), clip=5.0)
running_state.rs._M = np.array([ 1.09601490e-01, -4.17194803e-01,  5.11426850e-01,  1.44000289e-01, \
 -7.86010730e-01, -1.49164759e-01,  2.72771682e-03, -1.36839593e-02, \
  1.31160604e-02, -3.14483549e-02, -1.38137464e-02, -7.79511745e-03, \
  1.28112518e+00,  9.00026252e-01,  5.09637324e+00, -7.15813110e-04, \
 -1.75157281e-04,  6.82031951e-05,  1.28492229e+00,  9.04737207e-01, \
  5.09710362e+00])
network = ActorCritic(num_inputs, num_actions, layer_norm=True)
network.load_state_dict(torch.load("./RL_algorithms/Torch/result/actor_agent4.pt"))

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.

state = env.reset()["observation"]
state = running_state(state, update = False)
reward_sum = 0
# frames = []
for step in range(100):
    env.render()
    # frame = env.render("rgb_array")
    # frames.append(frame)
    network.eval()
    action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
    action, logproba = network.select_action(action_mean, action_logstd)
    action = action.data.numpy()[0]
    logproba = logproba.data.numpy()[0]
    next_state, reward, done, _ = env.step(action)
    next_state = next_state["observation"]
    reward_list.append(reward)
    reward_sum += reward
    next_state = running_state(next_state, update = False)
    mask = 0 if done else 1
    state = next_state
    if done:
        break
    print("action: ",action,"\n observation: ",next_state)
print(reward_sum)
# imageio.mimsave(str(parent_dir) + "/render.gif",frames,duration=0.1)
env.close()