import gym
from gym import spaces
import tensorflow as tf
# import stable_baselines
import numpy as np
import itertools
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


# import warnings
# warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
# warnings.filterwarnings("ignore", message=r"WARNING", category=FutureWarning)
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging 
logging.getLogger('tensorflow').disabled = True
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # surpress warning
# tf.get_logger().setLevel('ERROR')
TON2MG = 0.907185


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
            


class FCEnv(gym.Env):  # the forest carbon env
    metadata = {'render.modes': ['human']}

    def __init__(self, K_T=0.019, K_S=1/6.2, K_P=1/100, r_1=1/37, d_t=656, NPP_y=2.5, NPP_o=0.2):
        # parameters = {=
        #     "K_T": 0.019, # tree litterfall rate
        #     "K_S": 1/6.2, # soil decay
        #     "K_P": 1/100, # product decay
        #     "r_1": 1/37, # product decay
        #     "d_t" : 1, # tree density
        #     "NPP_y": 2.5, # carbon released tons/ha/year for young trees
        #     "NPP_o": 0.2, # carbon released tons/ha/year for old trees
        # }
        self.age_rate = 0.05 #fraction of young trees converted to old trees
        self.growth_rate = 1.025 #rate of increase in number of (young) trees
        self.litter_rate = 0.01 #fraction of tree C converted to soil C
        self.soil_decay = 0.02 #k constant for soil decay first-order eq
        self.product_decay = 0.01 #k constant for product decay first-order eq

        self.K_T = K_T
        self.K_S = K_S
        self.K_P = K_P
        self.r_1 = r_1
        self.d_t = d_t
        self.NPP_y = NPP_y
        self.NPP_o = NPP_o
        self.eco_step_matrix = np.array(
                [[-1*K_S, K_T, 0, 0, 0], # soil_carbon
                 [0, -1*K_T, 0, 0, 0],   # tree_carbon
                 [0, 0, -1*K_P, 0, 0],   # product_carbon
                 [0, 0, 0, 0, r_1],    # oldtree_ct
                 [0, 0, 0, 0, -1*r_1]] # youngtree_ct
            ) 
        # action: fraction of old trees removed
        self.action_space = gym.spaces.Discrete(101) # 0-1.00 % of old trees to cut down
        # self.action_space = spaces.Box(np.array([1.]), np.array([2]), dtype=np.float32) # porportion of old trees to cut down
            # Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32) # [-1,2] for first dimension and [-2,4] for second dimension 

        # state/observation = [soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct] (tree species, size/age classes)
        low = np.zeros( (5,), dtype=np.float32)
        high = np.array([float('inf')]*5, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None # 6 element np array

    def reset(self):
        """Returns state value in self.observation_space. Re-start the environment. """
        self.state = np.array([150, 76, 0, 328, 328], dtype=np.float32)
        return self.state

    def step(self, action):
        """argument action is within action_space (integer or numpy array) 
           action: porportion of old trees to make into product)
        """
        # assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = self.state # State at t (action picked after 00 updates)
        orig_total_carbon = tree_carbon + soil_carbon + product_carbon

        #1. Apply environmental carbon update to state_t1 (differential equations)
        self.eco_step(1)

        frac_tree_cut = max(min(action/100, 1), 0)
        num_tree_cut = oldtree_ct*frac_tree_cut
        oldtree_ct = max(0, oldtree_ct - num_tree_cut)
        tree_carbon = max(0, tree_carbon-num_tree_cut*0.1) # each tree has 0.1 Mg C/ha
        product_carbon += num_tree_cut*0.2 # good to cut trees
        
        carbon_sequestered = tree_carbon+soil_carbon+product_carbon - orig_total_carbon
        # print(f"\taction={action}, frac_tree_cut={frac_tree_cut}, num_tree_cut={num_tree_cut}, carbon_sequestered={carbon_sequestered}")

        reward = 1.0 * carbon_sequestered #can have different reward function

        done = bool(youngtree_ct + oldtree_ct == 0.0)
    
        # -> At t+1, we see state_t1+ carbon_update + action_update
        
        self.state = np.array([soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct], dtype=np.float32)
        return self.state, reward, done, {}

    def eco_step(self, num_steps = 1):
        """Updates self.state based on differential equation model of the ecosystem of forest and carbon """

        for i in range(num_steps):
            soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = self.state 
            f = TON2MG * (youngtree_ct/self.d_t * self.NPP_y + oldtree_ct/self.d_t * self.NPP_o) # tree growth's intake of carbon
            g = 0.1* youngtree_ct + 0.2*oldtree_ct # reproduction of trees
            print(f"eco_step: self.state={self.state}, f={f}, g={g},")
            self.state += np.matmul(self.eco_step_matrix, self.state) + np.array([0, f, 0, 0, g])  # 2d and 1d  # (5,)
            print(f"eco_step: self.state={self.state}")

    


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def dqn(parameters, undisturbed = False):
    # https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py
    with U.make_session(num_cpu=8):
        env = FCEnv(**parameters) # Create the environment
        # state_space_size = env.observation_space.shape # (5,)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name), # input placeholder for specific observation space
            q_func=model,
            num_actions=env.action_space.n,  # (1,)
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        replay_buffer = ReplayBuffer(50000) # Create the replay buffer, Max number of transitions to store in the buffer
        # Schedule for exploration: 1 (every action is random) -> 0.02 (98% of actions selected by values predicted by model)
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=0.00, final_p=0.00)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()
        episode_rewards = []
        stepct = 0
        for episode in range(num_episodes):
            episode_rewards.append(0)  # each ele = reward from one episode
            state = env.reset()  # returns state
            done = False
            
            for step in range(steps_per_episode):
                stepct+=1
                if undisturbed:
                    env.eco_step(10)
                    print(f"Episode-step {episode}-{step}: env.state={env.state}")
                    continue # to next step directly
                env.eco_step(99)
                # Pick action and update exploration to the newest value
                action = act(state[None], stochastic=False, update_eps=exploration.value(stepct))[0] # observation obj (axis added), stochastic boolean, update
                new_state, rew, done, info = env.step(action)
                print(f"Episode-step {episode}-{step}: new_state={new_state}")
                # Store transition in the replay buffer.
                replay_buffer.add(state, action, rew, new_state, float(done))
                state = new_state
                episode_rewards[-1] += rew
                if done:  break # go to next episode
                if episode > -1: # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32) # Sample a batch of experiences
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

            # end of an episode
            update_target() # Update target network periodically.
            logger.record_tabular("steps", stepct)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(stepct)))
            logger.dump_tabular()


if __name__ == '__main__':
    num_episodes = 1 # update once per episode
    steps_per_episode = 1
    # learning_rate = 0.1
    # discount_rate = 0.99
    # exploration_rate = 1
    # max_exploration_rate = 1
    # min_exploration_rate = 0.01
    # exploration_decay_rate = 0.01 #if we decrease it, will learn slower
    # rewards_all_episodes = []
    parameters = {
        "K_T": 0.019, # tree litterfall rate
        "K_S": 1/6.2, # soil decay
        "K_P": 1/100, # product decay
        "r_1": 1/37, # product decay
        "d_t" : 656, # tree density
        "NPP_y": 2.5, # carbon released tons/ha/year for young trees
        "NPP_o": 0.2, # carbon released tons/ha/year for old trees
    }
    dqn(parameters, undisturbed=True)


##### NOTES
# act: function that choses an action given an observation
# train: function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:
    # td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
    # loss = huber_loss[td_error]
