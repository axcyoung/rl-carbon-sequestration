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

import json
import time

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
            


class FCEnv(gym.Env):  # the forest carbon env
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_state=np.array([150, 76, 0, 328, 328], dtype=np.float32), 
                K_T=0.019, K_S=1/6.2, K_P=1/100, m_1=1/37, d_t=656, NPP_y=2.5, NPP_o=0.2, 
                r_o=0.6, r_y=0.2, C_o = 0.076, C_y = 0.038, death_o = 0.05, death_y=0.1):
        self.initial_state = np.copy(initial_state)
        self.K_T = K_T # tree litterfall rate
        self.K_S = K_S # soil decay
        self.K_P = K_P  # product decay
        self.m_1 = m_1 # maturation rate (young to old)
        self.d_t = d_t # tree density
        self.NPP_y = NPP_y # carbon released tons/ha/year for young trees
        self.NPP_o = NPP_o # carbon released tons/ha/year for old trees
        self.r_o = r_o # reproduction rate for old
        self.r_y = r_y # reproduction rate for young
        self.C_o = C_o # amount of carbon per tree (Megagrams)
        self.C_y = C_y # amount of carbon per tree (Megagrams)
        self.death_o = death_o # death rate of old
        self.death_y = death_y # death rate of young

        self.eco_step_matrix = np.array(
                [[-1*K_S, K_T, 0, C_o*death_o, C_y*death_y], # soil_carbon
                 [0, -1*K_T, 0, -1*C_o*death_o, -1*C_y*death_y],   # tree_carbon
                 [0, 0, -1*K_P, 0, 0],   # product_carbon
                 [0, 0, 0, -1*death_o, m_1],    # oldtree_ct
                 [0, 0, 0, 0, -1*m_1+-1*death_y]] # youngtree_ct
            ) 
        print("eco_step_matrix:", self.eco_step_matrix)
        # action: fraction of old trees removed
        self.action_space = gym.spaces.Discrete(101) # 0-1.00 % of old trees to cut down
        # self.action_space = spaces.Box(np.array([1.]), np.array([2]), dtype=np.float32) # porportion of old trees to cut down
            # Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32) # [-1,2] for first dimension and [-2,4] for second dimension 

        # state/observation = [soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct] (tree species, size/age classes)
        low = np.zeros((5,), dtype=np.float32)
        high = np.array([float('inf')]*5, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None # 6 element np array

    def reset(self):
        """Returns state value in self.observation_space. Re-start the environment. """
        self.state = np.copy(self.initial_state)
        return self.state

    def step(self, action):
        """argument action is within action_space (integer or numpy array) 
           action: porportion of old trees to make into product)
        """
        # assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = self.state # State at t (action picked after 00 updates)
        orig_total_carbon = tree_carbon + soil_carbon + product_carbon

        # 1. Apply environmental carbon update to state_t1 (differential equations)
        self.eco_step(1)
        # 2. Apply action
        frac_tree_cut = max(min(action/100, 1), 0)
        num_tree_cut = oldtree_ct*frac_tree_cut
        oldtree_ct = max(0, oldtree_ct - num_tree_cut)
        tree_carbon = max(0, tree_carbon-num_tree_cut*self.C_o) # each tree has 0.1 
        product_carbon += num_tree_cut*self.C_o*1/2 # loses half

        carbon_sequestered = tree_carbon+soil_carbon+product_carbon - orig_total_carbon
        # print(f"\taction={action}, frac_tree_cut={frac_tree_cut}, num_tree_cut={num_tree_cut}, carbon_sequestered={carbon_sequestered}")

        reward = 1.0 * carbon_sequestered #can have different reward function

        done = bool(youngtree_ct + oldtree_ct == 0.0)
    
        # -> At t+1, we see state_t1+ carbon_update + action_update
        
        self.state = np.array([soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct], dtype=np.float32)
        return self.state, reward, done, {}

    def eco_step(self, num_steps = 1, delta = 0.01):
        """Updates self.state based on differential equation model of the ecosystem of forest and carbon """
        for i in range(num_steps):
            soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = self.state 
            f = TON2MG * (youngtree_ct/self.d_t * self.NPP_y + oldtree_ct/self.d_t * self.NPP_o) # tree growth's intake of carbon
            g = self.r_y*youngtree_ct + self.r_o*oldtree_ct # reproduction of trees
            # if (i % 100 == 0) or (i == num_steps-1):
            #     print(f"eco_step {i}: self.state={self.state}, f={f}, g={g}")
            self.state += delta * (np.matmul(self.eco_step_matrix, self.state) + np.array([0, f, 0, 0, g]))  # 2d and 1d  # (5,)
    
    def total_carbon(self):
        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = self.state
        return soil_carbon + tree_carbon + product_carbon



def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def eco_run(parameters, undisturbed_iter=1000, delta = 0.01):
    with U.make_session(num_cpu=8):
        env = FCEnv(**parameters) # Create the environment
        env.reset()
        env.eco_step(undisturbed_iter, delta)


def dqn2(parameters, max_episode_num = 10, steps_per_episode=100):
    """main difference in how things are looped. This one has max_episode_num and steps_per_episode.
        returns 'act': ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
     """
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
        act_params = {
            'make_obs_ph': lambda name: ObservationInput(env.observation_space, name=name),
            'q_func': model,
            'num_actions': env.action_space.n,
        }
        act = deepq.deepq.ActWrapper(act, act_params)

        replay_buffer = ReplayBuffer(50000) # Create the replay buffer, Max number of transitions to store in the buffer
        # Schedule for exploration: 1 (every action is random) -> 0.02 (98% of actions selected by values predicted by model)
        # exploration = LinearSchedule(schedule_timesteps=5000, initial_p=1, final_p=1)
        exploration_start = 0.9
        exploration_end = 0.02
        exploration_timestep = 2000
        exploration_change = (exploration_end-exploration_start)/exploration_timestep
        

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()
        exploration = exploration_start

        total_stepct = 0
        episode_rewards = [] # each ele = reward from one complete run until done
        actions_taken = []

        soil_carbons, tree_carbons, product_carbons, oldtree_cts, youngtree_cts = [], [], [], [], []
        for episode in range(max_episode_num): # each episode: restart run, keep learning on old weights
            state = env.reset()  # returns state
            episode_rewards.append(0)
            for step in range(steps_per_episode): # each step = 1 year, each episode = steps_per_episode years
                if episode == max_episode_num-1:
                    soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = env.state 
                    soil_carbons.append(soil_carbon)
                    tree_carbons.append(tree_carbon)
                    product_carbons.append(product_carbon)
                    oldtree_cts.append(oldtree_ct)
                    youngtree_cts.append(youngtree_ct)
                total_stepct+=1
                exploration = max(exploration + exploration_change, exploration_end)

                env.eco_step(99, delta = 0.01)
                if step%25==0: print(f"   [{episode}-{step}] preaction: state={env.state}, exploration={round(exploration, 5)}")
                ## Pick action and update exploration to the newest value
                # action = act(state[None], stochastic=False, update_eps=exploration.value(total_stepct))[0] # observation obj (axis added), stochastic boolean, update
                action = act(state[None], stochastic=False)[0] if np.random.uniform(0,1) > exploration else env.action_space.sample()
                new_state, rew, done, info = env.step(action) # updates state
                actions_taken.append(action)
                # if step%25==0: print(f"   \tpostaction: state={env.state}, action={action}")
                ## Store transition in the replay buffer.
                replay_buffer.add(state, action, rew, new_state, float(done))
                state = new_state
                episode_rewards[-1] += rew
                if done: break # go to next episode
                # is_solved = total_stepct > 100 and np.mean(episode_rewards[-101:-1]) >= 200
                # if is_solved:
                    # print(f"---SOLVED: {np.mean(episode_rewards[-101:-1])}---")
                    # env.render()
                else: 
                    if total_stepct > -1: # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32) # Sample a batch of experiences
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            # end of each episode
            update_target() # Update target network periodically.
            logger.record_tabular("steps", total_stepct)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            logger.record_tabular("% time spent exploring", int(exploration * 100)) #int(100 * exploration.value(total_stepct)
            logger.dump_tabular()
            print(f"### episode {episode} ### episode reward={episode_rewards[-1]} | Total carbon={env.total_carbon()}\n")

        endtime = time.time()
        print(soil_carbons)
        with open(f"output/{endtime}_last_episode_vals.json", 'w') as f:
            f.write(json.dumps({"soil_carbons":str(soil_carbons), "tree_carbons":str(tree_carbons), "product_carbons":str(product_carbons), 
                "oldtree_cts":str(oldtree_cts), "youngtree_cts":str(youngtree_cts)}, indent=4))
        episode_rewards = np.array(episode_rewards)/steps_per_episode
        # act.save(f"output/{endtime}_act.pkl")
        return endtime, actions_taken, episode_rewards, env.state


if __name__ == '__main__':
    start_time = time.time()
    parameters = {
        "initial_state": np.array([150, 76, 0, 328, 328], dtype=np.float32), # soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct
        "K_T": 0.008, # tree litterfall rate
        "K_S": 1/50, # soil decay
        "K_P": 1/10, # product decay
        "m_1": 1/37, # product decay
        "d_t" : 656, # tree density
        "NPP_y": 2.5, # carbon released tons/ha/year for young trees
        "NPP_o": 0.2, # carbon released tons/ha/year for old trees
        "r_o":0.6,
        "r_y":0.2,
        "C_o": 0.076, # 76
        "C_y": 0.038, # 38
        "death_o": 0.05, 
        "death_y": 0.1
    }

    # eco_run(parameters, undisturbed_iter=2000)
    # dqn1(parameters, max_step_ct = 1000)
    endtime, actions_taken, episode_rewards, ending_state = dqn2(parameters, max_episode_num = 10, steps_per_episode=100)
    print("\n------------------------ %s seconds ------------------------" % (endtime - start_time)) # round(time.time() - start_time, 2)
    print(f"actions_taken={actions_taken}, Number of actions_taken={len(actions_taken)}")
    print(f"episode_rewards={episode_rewards}")
    print(f"ending_state={ending_state}")
    print(parameters, "\n")
    with open("log.txt", 'a') as file:
        file.write(f"\n\n\n#####{endtime}#####\n")
        file.writelines(str(parameters))
        file.write("\n")
        file.writelines(str(np.array(actions_taken[-300:])))
        file.write("\n")
        file.writelines(str(episode_rewards))
        file.write("\n")
        file.writelines(str(ending_state))

        
    
    


##### NOTES
# act: function that choses an action given an observation
# train: function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:
    # td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
    # loss = huber_loss[td_error]
# https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/deepq/build_graph.py#L317


# def dqn1(parameters, max_step_ct=100):
#     # https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py
#     with U.make_session(num_cpu=32):
#         env = FCEnv(**parameters) # Create the environment
#         # Create all the functions necessary to train the model
#         act, train, update_target, debug = deepq.build_train(
#             make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name), # input placeholder for specific observation space
#             q_func=model,
#             num_actions=env.action_space.n,  # (1,)
#             optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
#             gamma = 0.99# relatively long-term
#         )
#         replay_buffer = ReplayBuffer(50000) # Create the replay buffer, Max number of transitions to store in the buffer
#         # Schedule for exploration: 1 (every action is random) -> 0.02 (98% of actions selected by values predicted by model)
#         exploration = LinearSchedule(schedule_timesteps=10000, initial_p=0.5, final_p=0.02)

#         # Initialize the parameters and copy them to the target network.
#         U.initialize()
#         update_target()

#         episode_rewards = [0.0] # each ele = reward from one complete run until done
#         state = env.reset()  # returns state
#         for stepct in itertools.count():
#             if stepct>=max_step_ct:
#                 return
#             env.eco_step(99)
#             if stepct % 25 == 0: print(f"[{stepct}] preaction: state={env.state}")
#             # Pick action and update exploration to the newest value
#             action = act(state[None], stochastic=False, update_eps=exploration.value(stepct))[0] # observation obj (axis added), stochastic boolean, update
#             new_state, rew, done, info = env.step(action)
#             if stepct % 25 == 0: print(f"\tpostaction: state={env.state}")
#             # Store transition in the replay buffer.
#             replay_buffer.add(state, action, rew, new_state, float(done))
#             state = new_state
#             episode_rewards[-1] += rew
#             if done:        
#                 state = env.reset()
#                 episode_rewards.append(0)
#             is_solved = stepct > 100 and np.mean(episode_rewards[-101:-1]) >= 200
#             if is_solved:
#                 print("------------SOLVED------------")
#                 # env.render()
#             else: 
#                 if stepct > -1: # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
#                     obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32) # Sample a batch of experiences
#                     train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
#                 if stepct % 100 == 0:
#                     update_target() # Update target network periodically.

#             # if done and len(episode_rewards) % 1 == 0: # every nth done reached
#             if stepct % 100 == 0: # every nth done reached
#                 logger.record_tabular("steps", stepct)
#                 logger.record_tabular("episodes", len(episode_rewards))
#                 logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
#                 logger.record_tabular("% time spent exploring", int(100 * exploration.value(stepct)))
#                 logger.dump_tabular()