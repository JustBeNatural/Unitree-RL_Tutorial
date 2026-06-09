# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg): # 类适合描述“一类对象”或“一组结构化数据”
    num_steps_per_env = 24  # 每个并行环境中每次收集的步骤数
    max_iterations = 50000  # 训练的最大迭代次数
    save_interval = 100     # 每隔多少迭代保存一次模型
    experiment_name = ""    # same as task name
    empirical_normalization = False # 是否使用经验归一化来标准化观察值和奖励值 # 这个参数可以让神经网络的数据尺度更稳定
    
    # 策略网络参数配置
    policy = RslRlPpoActorCriticCfg( # 这个配置的是Actor-Critic神经网络
        init_noise_std=1.0, # 初始动作噪声的标准差，较大的值可以促进探索
        # Actor网络控制的policy大脑的大小，critic网络控制的value大脑的大小
        actor_hidden_dims=[512, 256, 128], # Actor网络有三层隐藏层，分别有512、256和128个神经元
        critic_hidden_dims=[512, 256, 128], # Critic网络也有三层隐藏层，分别有512、256和128个神经元
        activation="elu",
    )
    # PPO算法参数配置
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
