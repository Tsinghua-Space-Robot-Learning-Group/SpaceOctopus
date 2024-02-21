import gym
import imageio
from PIL import Image, ImageDraw
import SpaceRobotEnv
import numpy as np
import torch
import sys
import os
from gym import spaces
import matplotlib.pyplot as plt

# os.environ["MUJOCO_GL"] = "egl"

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)
sys.path.append(parent_dir+"/RL_algorithms/Torch/MAPPO/onpolicy")
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.spacerobot.SpaceRobotFourArm_env import FourArm
from onpolicy.config import get_config

def _t2n(x):
    return x.detach().cpu().numpy()

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="SpaceRobotDualArmWithRot", 
                        help="which scenario to run on.")
    parser.add_argument("--num_agents", type=int, default=4,
                        help="number of agents.")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")
                        
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    env = FourArm(all_args)

    # env.seed(all_args.seed)
    # seed
    # torch.manual_seed(all_args.seed)
    # torch.cuda.manual_seed_all(all_args.seed)
    # np.random.seed(all_args.seed)

    eval_episode_rewards = []
    actors = []
    obs = env.reset()

    eval_rnn_states = np.zeros((1),dtype=np.float32)
    eval_masks = np.ones((1), dtype=np.float32)

    for i in range(all_args.num_agents):
        act = R_Actor(all_args,env.observation_space[i],env.action_space[i])
        act.load_state_dict(torch.load("/home/zhaowenbo/Documents/MARL_SpaceRobot/RL_algorithms/Torch/MAPPO/onpolicy/scripts/results/SpaceRobotEnv/SpaceRobotFourArm/mappo/EightAgents/run1/models/actor_agent"+str(i)+".pt"))
        actors.append(act)
        # print(act.act.action_out.logstd._bias)

    frames = []
    print("init goal:", env.env.goal)
#     env.env.goal = np.array([
#     0.29432666,  1.3777534,   4.3778887,   0.16373104,  0.23992744,  0.05128532,
#     0.4315827,  -1.2804029,   4.2666655,   0.03040494,  0.11790288, -0.04894744,
#     0.27814835,  0.29937622,  5.246062,   -1.0688201,   0.339479,    0.16324279,
#     0.43091363,  0.28066614,  2.7131298,  -0.89631516,  0.00982002, -0.03541239
# ])
#     print("changed goal: ", env.env.goal)
    
    x = np.linspace(0, 19.9, 200)
    y = np.zeros([8, 200])


    with torch.no_grad():
        # print(env.env.initial_gripper1_pos,env.env.initial_gripper1_rot,env.env.initial_gripper2_pos,env.env.initial_gripper2_rot)
        env_goal = env.env.goal
        for eval_step in range(all_args.episode_length):
            print("step: ",eval_step)
            img = env.env.render("rgb_array")
            # env.env.render()
            frames.append(img)
            action = []
            for agent_id in range(all_args.num_agents):
                actor = actors[agent_id]
                actor.eval()
                # print(actor.act.action_out.logstd._bias)
                # print("observation: ",np.array(list(obs[agent_id,:])).reshape(1,31),)
                ob_i = np.array(list(obs[agent_id,:])).reshape(31,)
                error_i = goal_distance(ob_i[19:22], env_goal[agent_id * 3: (agent_id + 1) * 3])
                y[agent_id, eval_step] = error_i
                eval_action,_,rnn_states_actor = actor(
                    np.array(list(obs[agent_id,:])).reshape(1,31),
                    eval_rnn_states,
                    eval_masks,
                    deterministic=True,
                )
                eval_action = eval_action.detach().cpu().numpy()
                # print("step: ",eval_step,"action: ",eval_action)
                action.append(eval_action)

            obs, eval_rewards, done, infos = env.step(np.stack(action).squeeze().reshape(all_args.num_agents,3))
            print("reward: ",eval_rewards)
            # print("action: ",np.stack(action).squeeze().reshape(all_args.num_agents,3))
            eval_episode_rewards.append(eval_rewards)
        print("episode reward: ",np.array(eval_episode_rewards).sum())
        
    print("goal:, ", env.env.goal)
    for i in range(8):
        plt.plot(x, y[i,:], label = "error %d" %i)
    plt.legend()
    plt.savefig('sim1.png')

    imageio.mimsave(
        str(parent_dir) + "/render.gif",
        frames,
        duration=0.01,
    )
        
        # writer = imageio.get_writer(parent_dir + "/render.gif")
        # # print('reward is {}'.format(self.reward_lst))
        # for frame, reward in zip(frames, eval_episode_rewards):
        #     print(eval_step)
        #     frame = Image.fromarray(frame)
        #     draw = ImageDraw.Draw(frame)
        #     draw.text((70, 70), '{}'.format(reward), fill=(255, 255, 255))
        #     frame = np.array(frame)
        #     writer.append_data(frame)
        # writer.close()
        # env.close()

if __name__ == "__main__":
    main(sys.argv[1:])