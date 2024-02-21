import gym

import SpaceRobotEnv
import numpy as np

env = gym.make("SpaceRobotBaseRot-v0")
print(env.sim.data.qpos)
# print(env.initial_state)
action = np.zeros(24,)
# action[4] = 1
env.sim.data.ctrl[:] = action
env.sim.step()
print(env.sim.data.qpos)
print(env.sim.data.qvel)

dim_u = env.action_space.shape[0]
print(dim_u)
dim_o = env.observation_space["observation"].shape[0]
print(dim_o)


# print(env.action_space)
# print(env.observation_space["observation"])

# bounds = env.model.actuator_ctrlrange.copy()
# low, high = bounds.T

# print(low[0:3],high)

observation = env.reset()
ob = observation["observation"]

max_action = env.action_space.high
print("max_action:", max_action)
print("min_action", env.action_space.low)
for e_step in range(1):
    # print(env.initial_gripper1_pos,env.initial_gripper2_pos,env.initial_gripper1_rot,env.initial_gripper2_rot)
    observation = env.reset()
    print(e_step, env.goal)
    for i_step in range(50):
        # env.render()
        img = env.render("rgb_array")
        # action = np.random.uniform(low=-1.0, high=1.0, size=(dim_u,))
        observation, reward, done, info = env.step(max_action * action)
        # print(observation["observation_1"])
        if i_step == 0:
            print(reward)
            from PIL import Image 
            adv = Image.fromarray(np.uint8(img))
            adv.save("fig.jpg", quality = 95)
env.close()
