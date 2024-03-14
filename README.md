# SpaceRobotEnv

> Note: our repo can be found in the OpenAI Gym Documentation now. Please see [SpaceRobotEnv](https://www.gymlibrary.dev/environments/third_party_environments/#spacerobotenv).    

The SpaceOctopus environment is built upon our previous work [SpaceRobotEnv](https://github.com/Tsinghua-Space-Robot-Learning-Group/SpaceRobotEnv). SpaceRobotEnv is an open-sourced environments for trajectory planning of free-floating space robots.
Different from the traditional robot, the free-floating space robot is a dynamic coupling system because of the non-actuated basew. 
Therefore, model-based trajectory planning methods encounter many dif- ficulties in modeling and computing. 
SpaceRobotEnv are developed with the following key features:
* **Real Space Environment**: we construct environments similar to the space. The free-floating space robot is located in a low-gravity condition.
* **Dynamic coupling control**: Compared with robots on the ground, the torques of joints have a significant impact on the posture of the base. The movement of the base makes a disturbance on the positions of end-effectors, thus leading to a more complex trajectory planning task. 
* **Image input**: We provide the ability to use images as observations. And we also demonstrates our environment is effective, please see [our paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9550509).   

The paper of SpaceOctopus can be found [here](https://arxiv.org/abs/2403.08219). In the paper, we observe that the octopus can elegantly conduct trajectory planning while adjusting its pose during grabbing prey or escaping from danger. 
Inspired by the distributed control of octopuses' limbs, we develop a multi-level decentralized motion planning framework to manage the movement of different arms of space robots. 
This motion planning framework integrates naturally with the multi-agent reinforcement learning (MARL) paradigm. 
The results indicate that our method outperforms the previous method (centralized training). Leveraging the flexibility of the decentralized framework, we reassemble policies trained for different tasks, enabling the space robot to complete trajectory planning tasks while adjusting the base attitude without further learning. 
Furthermore, our experiments confirm the superior robustness of our method in the face of external disturbances, changing base masses, and even the failure of one arm.

## Installation

Our environment is built on the [Mujoco Simulation](https://github.com/deepmind/mujoco). So before using our repo, please make sure you install the [Mujoco](https://github.com/deepmind/mujoco) platform.
Additionally, our framework is based on the [Gym](https://github.com/openai/gym).
Details regarding installation of Gym can be found [here](https://github.com/openai/gym).

After you finish the installation of the Mujoco and Gym and test some toy examples using them, you can install this repo from the source code:

```bash
pip install -e .
```

Further, you also have to cd to the /onpolicy folder and run the same command to install the MAPPO package:
```bash
pip install -e .
```
More information about the MAPPO algorithm and the installation details can be found in the [original repo](https://github.com/marlbenchmark/on-policy).

## Quick Start

We provide a Gym-Like API that allows us to get interacting information. `test_env.py` shows a toy example to verify the environments.
As you can see, A Gym-Like API makes some popular RL-based algorithm repos, like [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), easily implemented in our environments.
```python
import gym

import SpaceRobotEnv
import numpy as np

env = gym.make("SpaceRobotBaseRot-v0")

dim_u = env.action_space.shape[0]
print(dim_u)
dim_o = env.observation_space["observation"].shape[0]
print(dim_o)


observation = env.reset()
max_action = env.action_space.high
print("max_action:", max_action)
print("mmin_action", env.action_space.low)
for e_step in range(20):
    observation = env.reset()
    for i_step in range(50):
        env.render()
        action = np.random.uniform(low=-1.0, high=1.0, size=(dim_u,))
        observation, reward, done, info = env.step(max_action * action)

env.close()
```

## Introduction of multi-arm space robot

In the multi-arm space robot setting, four 6-degree-of-freedom (6-DoF) UR5 robotic arms are rigidly attached to the base of the space robot, with parameters identical to those of the actual robot. In the trajectory planning task, the goal for each end-effector is to reach a target randomly selected from an area within a 0.3 $\times$ 0.3 $\times$ 0.3 $\mathrm{m}^3$ cube positioned in front of each arm, along with a randomly sampled desired orientation. For the base reorientation task, the desired base attitude is randomly determined, ranging from -0.2 rad to 0.2 rad along every axis. The mass of the base is 400 kg with its size  0.8726 $\times$ 0.8726 $\times$ 0.8726 $\mathrm{m}^3$. Assuming the gripper of the robotic arm is insensitive to the shape of the object, we disregard the shape of grippers. Additionally, we omit the modeling of solar panels due to their negligible impact on the base, and the entire system is unaffected by gravity.

<div align=center>
<img src="render/fig.jpg" align="center" width="600"/>
</div> 


## Citing SpaceRobotEnv

If you find SpaceRobotEnv useful, please cite our recent work in your publications. 

```
@misc{zhao2024spaceoctopus,
      title={SpaceOctopus: An Octopus-inspired Motion Planning Framework for Multi-arm Space Robot}, 
      author={Wenbo Zhao and Shengjie Wang and Yixuan Fan and Yang Gao and Tao Zhang},
      year={2024},
      eprint={2403.08219},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}

@article{wang2022collision,
  title={Collision-Free Trajectory Planning for a 6-DoF Free-Floating Space Robot via Hierarchical Decoupling Optimization},
  author={Wang, Shengjie and Cao, Yuxue and Zheng, Xiang and Zhang, Tao},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={4953--4960},
  year={2022},
  publisher={IEEE}
}

@inproceedings{wang2021multi,
  title={A Multi-Target Trajectory Planning of a 6-DoF Free-Floating Space Robot via Reinforcement Learning},
  author={Wang, Shengjie and Zheng, Xiang and Cao, Yuxue and Zhang, Tao},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3724--3730},
  organization={IEEE}
}

@inproceedings{wang2021end,
  title={An End-to-End Trajectory Planning Strategy for Free-floating Space Robots},
  author={Wang, Shengjie and Cao, Yuxue and Zheng, Xiang and Zhang, Tao},
  booktitle={2021 40th Chinese Control Conference (CCC)},
  pages={4236--4241},
  year={2021},
  organization={IEEE}
}

@article{cao2022reinforcement,
  title={Reinforcement Learning with Prior Policy Guidance for Motion Planning of Dual-Arm Free-Floating Space Robot},
  author={Cao, Yuxue and Wang, Shengjie and Zheng, Xiang and Ma, Wenke and Xie, Xinru and Liu, Lei},
  journal={arXiv preprint arXiv:2209.01434},
  year={2022}
}

```  
  
## The Team

SpaceRobotEnv is a project maintained by 
[Shengjie Wang](https://github.com/Shengjie-bob), [Xiang Zheng](https://github.com/x-zheng16), [Yuxue Cao](https://github.com/ShenGe123000) , [Fengbo Lan](https://github.com/lanrobot), [Wenbo Zhao](https://github.com/Githuber-zwb)  at Tsinghua University. Also thanks a lot for the great contribution from [Tosin](https://github.com/tohsin)  .


## License

SpaceRobotEnv has an Apache license, as found in the [LICENSE](LICENSE) file.
