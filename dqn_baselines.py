import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def dqn1(parameters, max_step_ct=100):
    # https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py
    with U.make_session(num_cpu=32):
        env = FCEnv(**parameters) # Create the environment
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name), # input placeholder for specific observation space
            q_func=model,
            num_actions=env.action_space.n,  # (1,)
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            gamma = 0.99# relatively long-term
        )
        replay_buffer = ReplayBuffer(50000) # Create the replay buffer, Max number of transitions to store in the buffer
        # Schedule for exploration: 1 (every action is random) -> 0.02 (98% of actions selected by values predicted by model)
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=0.5, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0] # each ele = reward from one complete run until done
        state = env.reset()  # returns state
        for stepct in itertools.count():
            if stepct>=max_step_ct:
                return
            env.eco_step(99)
            if stepct % 25 == 0: print(f"[{stepct}] preaction: state={env.state}")
            # Pick action and update exploration to the newest value
            action = act(state[None], stochastic=False, update_eps=exploration.value(stepct))[0] # observation obj (axis added), stochastic boolean, update
            new_state, rew, done, info = env.step(action)
            if stepct % 25 == 0: print(f"\tpostaction: state={env.state}")
            # Store transition in the replay buffer.
            replay_buffer.add(state, action, rew, new_state, float(done))
            state = new_state
            episode_rewards[-1] += rew
            if done:        
                state = env.reset()
                episode_rewards.append(0)
            is_solved = stepct > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                print("------------SOLVED------------")
                # env.render()
            else: 
                if stepct > -1: # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32) # Sample a batch of experiences
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                if stepct % 100 == 0:
                    update_target() # Update target network periodically.

            # if done and len(episode_rewards) % 1 == 0: # every nth done reached
            if stepct % 100 == 0: # every nth done reached
                logger.record_tabular("steps", stepct)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(stepct)))
                logger.dump_tabular()

if __name__ == '__main__':
    with U.make_session(num_cpu=8):
        # Create the environment
        # env = gym.make("CartPole-v0")
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
