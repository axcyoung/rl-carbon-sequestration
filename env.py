import gym
from gym import spaces
import tensorflow as tf
import stable_baselines
import numpy as np



import itertools
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


def q_learning_discrete():
    ## Q-Learning algorithm: https://github.com/MatePocs/gym-basic/blob/main/gym_basic_env_test.ipynb
    env = BasicEnv()
    action_space_size = env.action_space.shape # (1,)
    state_space_size = env.observation_space.shape # (5,)
    q_table = np.zeros((5, 1))
    print(q_table)

    num_episodes = 1
    max_steps_per_episode = 10 # but it won't go higher than 1?
    learning_rate = 0.1
    discount_rate = 0.99
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
            


class BasicEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.age_rate = 0.05 #fraction of young trees converted to old trees
    self.growth_rate = 1.025 #rate of increase in number of (young) trees
    self.litter_rate = 0.01 #fraction of tree C converted to soil C
    self.soil_decay = 0.02 #k constant for soil decay first-order eq
    self.product_decay = 0.01 #k constant for product decay first-order eq
    
    #fraction of old trees removed
    self.action_space = spaces.Box(np.array([0.]), np.array([1.]), dtype=np.float32) # porportion of old trees to cut down
    # Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32) # [-1,2] for first dimension and [-2,4] for second dimension 

    # youngtree_count, oldtree_count, tree_carbon, soil_carbon, product_carbon (tree species, size/age classes)
    low = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
    high = np.array([float('inf'), float('inf'), float('inf'), float('inf'), float('inf')], dtype=np.float32)
    self.observation_space = spaces.Box(low, high, dtype=np.float32) # TODO
    self.state = None 

  def reset(self):
    """Returns state value in self.observation_space. Re-start the environment. """
    self.state = [656., 76., 150., 0., 0]
    return np.array(self.state, dtype=np.float32)

  def step(self, action):
    """argument action is within action_space (integer or numpy array) """
    err_msg = f"{action!r} ({type(action)}) invalid"
    assert self.action_space.contains(action), err_msg
    assert self.state is not None, "Call reset before using step method."

    youngtree_count, oldtree_count, tree_carbon, soil_carbon, product_carbon = self.state
    original_total_carbon = tree_carbon + soil_carbon + product_carbon

    #insert differential equations here

    self.state = [youngtree_count, oldtree_count, tree_carbon, soil_carbon, product_carbon]

    done = bool(youngtree_count + oldtree_count != 0.0) #simulation over?
    carbon_sequestered = tree_carbon + soil_carbon + product_carbon - original_total_carbon
    reward = 1.0 * carbon_sequestered #can have different reward function
    
    return np.array(self.state, dtype=np.float32), reward, done, {}


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def dqn():
    # https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py
    with U.make_session(num_cpu=8):
        env = BasicEnv() # Create the environment
        # env = gym.make("CartPole-v0")
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name), # input placeholder for specific observation space
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        replay_buffer = ReplayBuffer(50000) # Create the replay buffer
        # Schedule for exploration: 1 (every action is random) -> 0.02 (98% of actions selected by values predicted by model)
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        state = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(state[None], stochastic=False, update_eps=exploration.value(t))[0] # observation object, stochastic boolean, update
            new_state, rew, done, info = env.step(action)

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_state, float(done))
            obs = new_state

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            # is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if done: # is_solved
                env.render()
            else: # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
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


if __name__ == '__main__':
    dqn()


##### NOTES
# act: function that choses an action given an observation
# train: function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:
    # td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
    # loss = huber_loss[td_error]
