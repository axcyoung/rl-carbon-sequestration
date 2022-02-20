import gym
from gym import spaces
import tensorflow as tf
import numpy as np
import os
import itertools
import json
import time
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U



def q_learning_discrete():
    ## Q-Learning algorithm: https://github.com/MatePocs/gym-basic/blob/main/gym_basic_env_test.ipynb
    env = FCEnv()
    action_space_size = env.action_space.shape # (1,)
    state_space_size = env.observation_space.shape # (5,)
    q_table = np.zeros((5, 1))
    print(q_table)

    num_episodes = 1
    max_steps_per_episode = 10 # but it won't go higher than 1?
    learning_rate = 0.1
    discount_rate = 0.9
    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01 #if we decrease it, will learn slower
    rewards_all_episodes = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        rewards_current_episode = 0
        for step in range(max_steps_per_episode):
            exploration_rate_threshold = np.random.uniform(0,1) # Exploration -exploitation trade-off
            action = np.argmax(q_table[state,:]) if exploration_rate_threshold > exploration_rate  else env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            print(f"{step}: {new_state}, {reward}, {done}, {info}")
            # Update Q-table for Q(s,a)
            q_table[state, action] = (1-learning_rate) * q_table[state, action] + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
            state = new_state
            rewards_current_episode += reward
            if done == True: break  
        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

        rewards_all_episodes.append(rewards_current_episode)

    avg_episode_num = 1
    avg_rewards = np.split(np.array(rewards_all_episodes), num_episodes / avg_episode_num)
    count = 0
    print("********** Average  reward per thousand episodes **********\n")
    for r in avg_rewards:
        print(count, ": ", str(sum(r / avg_episode_num)))
        count += avg_episode_num
        
    # Print updated Q-table
    print("\n\n********** Q-table **********\n", q_table)
            

