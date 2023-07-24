import gym
import imageio
from PIL import Image, ImageDraw
import SpaceRobotEnv
import numpy as np
import torch
import sys
import os
from gym import spaces

# os.environ["MUJOCO_GL"] = "egl"

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)
sys.path.append(parent_dir+"/RL_algorithms/Torch/MAPPO/onpolicy")
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from envs.spacerobot.SpaceRobotDualArmOnlyPos_Env import DualArmWithRot
from config import get_config

def _t2n(x):
    return x.detach().cpu().numpy()

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

    env = DualArmWithRot(all_args)
    eval_episode_rewards = []
    actors = []
    obs = env.reset()

    eval_rnn_states = np.zeros((1),dtype=np.float32)
    eval_masks = np.ones((1), dtype=np.float32)

    for i in range(all_args.num_agents):
        act = R_Actor(all_args,env.observation_space[i],env.action_space[i])
        act.load_state_dict(torch.load("./RL_algorithms/Torch/MAPPO/onpolicy/scripts/results/SpaceRobotEnv/SpaceRobotDualArmWithRot/mappo/check/run23/models/actor_agent"+str(i)+".pt"))
        actors.append(act)

    with torch.no_grad():
        # print(env.env.initial_gripper1_pos,env.env.initial_gripper1_rot,env.env.initial_gripper2_pos,env.env.initial_gripper2_rot)
        for eval_step in range(all_args.episode_length):
            print("step: ",eval_step)
            # env.env.render()
            action = []
            for agent_id in range(all_args.num_agents):
                actor = actors[agent_id]
                actor.eval()
                eval_action,_,rnn_states_actor = actor(
                    np.array(list(obs[agent_id,:])).reshape(1,28),
                    eval_rnn_states,
                    eval_masks,
                    deterministic=True,
                )
                eval_action = eval_action.detach().cpu().numpy()
                # print(eval_step,eval_action)
                action.append(eval_action)

            obs, eval_rewards, done, infos = env.step(np.stack(action).squeeze().reshape(all_args.num_agents,3))
            eval_episode_rewards.append(eval_rewards)
        
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