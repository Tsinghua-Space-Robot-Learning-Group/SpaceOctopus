#!/bin/sh
# exp param
env="SpaceRobotEnv"
# scenario="SpaceRobotBaseRot"
scenario="SpaceRobotBaseRot"
algo="mappo" # "rmappo" "ippo"
exp="EightAgents"

# SpaceRobot param
# num_agents=4
num_agents=8
seed=2
seed_max=3

# train param
num_env_steps=20000000 #20M
episode_length=50

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, num agents is ${num_agents}, max seed is ${seed_max}"
# for seed in `seq ${seed_max}`;
# do
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python RL_algorithms/Torch/MAPPO/onpolicy/scripts/train/train_spacerobot.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--n_rollout_threads 128  --use_wandb --share_policy --share_reward --hidden_size 512 --layer_N 2 --ppo_epoch 5 \
--save_interval 20 --log_interval 5 --entropy_coef 0.05 --lr 8e-4 --critic_lr 8e-4
# --use_valuenorm --use_popart --gamma 0.96 --use_policy_active_masks --use_value_active_masks  --use_ReLU  \
# --entropy_coef 0.0 --lr 1e-3 --critic_lr 1e-3 \
# --use_valuenorm \
# --use_popart --share_reward 
# --lr 1e-3 --critic_lr 1e-3 \
# --use_feature_normalization --use_orthogonal \
# done
