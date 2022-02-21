from cgitb import reset
import gym
from gym import spaces
import tensorflow as tf
import numpy as np
import os
import itertools
import json
import time
import math
import matplotlib.pyplot as plt
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
# import logging 
# logging.getLogger('tensorflow').disabled = True
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # surpress warning
# tf.get_logger().setLevel('ERROR')

TON2MG = 0.907185

class FCEnv(gym.Env):  # the forest carbon env
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_state=np.array([150, 76, 0, 328, 328, 76, 328, 328], dtype=np.float32),
                litterfall_rate=0.019, soil_decay=1/6.2, product_decay=1/100, tree_mature_rate=1/37, tree_mature_rate2=1/37, tree_density=656, 
                tree_carbon_y_tonhayear=2.5, tree_carbon_o_tonhayear=0.2, tree_carbon_y_tonhayear2=2.5, tree_carbon_o_tonhayear2=0.2, 
                reproduction_o=0.6, reproduction_y=0.2, reproduction_o2=0.6, reproduction_y2=0.2, 
                tree_carbon_o_mgpertree = 0.076, tree_carbon_y_mgpertree = 0.038, tree_carbon_o_mgpertree2 = 0.076, tree_carbon_y_mgpertree2 = 0.038, 
                tree_death_o = 0.05, tree_death_y=0.1, tree_death_o2 = 0.05, tree_death_y2=0.1, carrying_capacity=10000):
        self.initial_state = np.copy(initial_state)
        self.litterfall_rate = litterfall_rate # tree litterfall rate
        self.soil_decay = soil_decay # soil decay
        self.product_decay = product_decay  # product decay
        self.tree_mature_rate = tree_mature_rate # maturation rate (young to old)
        self.tree_density = tree_density # tree density
        self.tree_carbon_y_tonhayear = tree_carbon_y_tonhayear # carbon released tons/ha/year for young trees
        self.tree_carbon_o_tonhayear = tree_carbon_o_tonhayear # carbon released tons/ha/year for old trees
        self.reproduction_o = reproduction_o # reproduction rate for old
        self.reproduction_y = reproduction_y # reproduction rate for young
        self.tree_carbon_o_mgpertree = tree_carbon_o_mgpertree # amount of carbon per old tree (Megagrams)
        self.tree_carbon_y_mgpertree = tree_carbon_y_mgpertree # amount of carbon per young tree (Megagrams)
        self.tree_death_o = tree_death_o # death rate of old
        self.tree_death_y = tree_death_y # death rate of young
        self.carrying_capacity = carrying_capacity

        self.tree_mature_rate2=tree_mature_rate2
        self.tree_carbon_o_tonhayear2=tree_carbon_o_tonhayear2
        self.tree_carbon_y_tonhayear2=tree_carbon_y_tonhayear2
        self.reproduction_o2=reproduction_o2
        self.reproduction_y2=reproduction_y2
        self.tree_carbon_o_mgpertree2=tree_carbon_o_mgpertree2
        self.tree_carbon_y_mgpertree2=tree_carbon_y_mgpertree2
        self.tree_death_o2=tree_death_o2
        self.tree_death_y2=tree_death_y2

        self.eco_step_matrix = np.array(
                [[-1*soil_decay, litterfall_rate, 0, tree_carbon_o_mgpertree*tree_death_o, tree_carbon_y_mgpertree*tree_death_y, litterfall_rate, tree_carbon_o_mgpertree2*tree_death_o2, tree_carbon_y_mgpertree2*tree_death_y2], # soil_carbon
                 [0, -1*litterfall_rate, 0, -1*tree_carbon_o_mgpertree*tree_death_o, -1*tree_carbon_y_mgpertree*tree_death_y, 0, 0, 0],   # tree_carbon
                 [0, 0, -1*product_decay, 0, 0, 0, 0, 0],   # product_carbon
                 [0, 0, 0, -1*tree_death_o, tree_mature_rate, 0, 0, 0],  # oldtree_ct
                 [0, 0, 0, 0, -1*tree_mature_rate+-1*tree_death_y, 0, 0, 0], # youngtree_ct
                 [0, 0, 0, 0, 0, -1*litterfall_rate,-1*tree_carbon_o_mgpertree2*tree_death_o2, -1*tree_carbon_y_mgpertree2*tree_death_y2], # tree_carbon2
                 [0, 0, 0, 0, 0, 0, -1*tree_death_o2, tree_mature_rate2], # oldtree_ct2
                 [0, 0, 0, 0, 0, 0, 0, -1*tree_mature_rate2+-1*tree_death_y2]] # youngtree_ct2
            ) # multiplied with state gives change to each state quanity
        eigenvalues, eigenvectors = np.linalg.eig(self.eco_step_matrix)
        print("eigenvalues:", eigenvalues)
        print("eigenvectors:", eigenvectors)
        print("eco_step_matrix:", self.eco_step_matrix)
        # Action: fraction of old trees removed
        self.action_space = gym.spaces.Discrete(21*21) # granularity=5 % of each speices

        # State/observation
        low = np.zeros((8,), dtype=np.float32)
        high = np.array([float('inf')]*8, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.state = None

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

        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct, tree_carbon2, oldtree_ct2, youngtree_ct2 = self.state # State at t (action picked after 00 updates)
        orig_total_carbon = tree_carbon + soil_carbon + product_carbon + tree_carbon2
        orig_abundance = oldtree_ct+youngtree_ct+oldtree_ct2+youngtree_ct2
        orig_biodiversity = compute_biodiversity(oldtree_ct, youngtree_ct, oldtree_ct2, youngtree_ct2)
        # 1. Apply environmental carbon update to state_t1 (differential equations)
        self.eco_step(1)
        # 2. Apply action
        frac_tree_cut = int(action/20)*5/100 
        frac_tree_cut2 = (action%20)*5/100
        num_tree_cut = oldtree_ct*frac_tree_cut
        num_tree_cut2 = oldtree_ct2*frac_tree_cut2
        oldtree_ct = max(0, oldtree_ct - num_tree_cut)
        oldtree_ct2 = max(0, oldtree_ct2 - num_tree_cut2)
        tree_carbon = max(0, tree_carbon-num_tree_cut*self.tree_carbon_o_mgpertree)
        tree_carbon2 = max(0, tree_carbon2-num_tree_cut2*self.tree_carbon_o_mgpertree2)
        product_carbon += num_tree_cut*self.tree_carbon_o_mgpertree*3/4 + num_tree_cut2*self.tree_carbon_o_mgpertree2*3/4# loses 25%
        # 3. Calculate rewards
        carbon_sequestered = tree_carbon+soil_carbon+product_carbon+tree_carbon2 - orig_total_carbon
        biodiversity_change = compute_biodiversity(oldtree_ct, youngtree_ct, oldtree_ct2, youngtree_ct2) - orig_biodiversity
        abundance_change = oldtree_ct+youngtree_ct+oldtree_ct2+youngtree_ct2 - orig_abundance
        # econ_profit
        reward = dqnparams["carbon_reward_weight"]*tanh(carbon_sequestered) +  \
                (dqnparams["biodiversiy_weight"])*tanh(biodiversity_change) + \
                (dqnparams["abundance_weight"])*tanh(abundance_change)

        done = bool(youngtree_ct + oldtree_ct == 0.0)
    
        # -> At t+1, we see state_t1 + eco_update + action_update
        self.state = np.array([soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct, tree_carbon2, oldtree_ct2, youngtree_ct2], dtype=np.float32)
        return self.state, reward, done, {}

    def eco_step(self, num_steps = 1, delta = 0.01, record=False):
        """Updates self.state based on differential equation model of the ecosystem of forest and carbon """
        if record:
            data = {"soil_carbons":[], "tree_carbons":[], "product_carbons":[], "oldtree_cts":[], "youngtree_cts":[], 
                "tree_carbons2":[], "oldtree_cts2":[], "youngtree_cts2":[]}
        for i in range(num_steps):
            soil_carbon, _, product_carbon, oldtree_ct, youngtree_ct, _, oldtree_ct2, youngtree_ct2 = self.state 
            carrying_capacity_coef = 1-(oldtree_ct+youngtree_ct+oldtree_ct2+youngtree_ct2)/self.carrying_capacity
            f = TON2MG * (youngtree_ct/self.tree_density*self.tree_carbon_y_tonhayear + oldtree_ct/self.tree_density*self.tree_carbon_o_tonhayear) # tree growth's carbon intake
            g = (self.reproduction_y*youngtree_ct+self.reproduction_o*oldtree_ct)*carrying_capacity_coef # reproduction of trees
            f2 = TON2MG * (youngtree_ct2/self.tree_density*self.tree_carbon_y_tonhayear2 + oldtree_ct2/self.tree_density*self.tree_carbon_o_tonhayear2) # tree growth's carbon intake
            g2 = (self.reproduction_y2*youngtree_ct2+self.reproduction_o2*oldtree_ct2)*carrying_capacity_coef # reproduction of trees
            # if (i % 100 == 0) or (i == num_steps-1):
            #     print(f"eco_step {i}: self.state={self.state}, f={f}, g={g}")
            self.state += delta * (np.matmul(self.eco_step_matrix, self.state) + np.array([0, f, 0, 0, g, f2, 0, g2]))  # 2d and 1d  # (5,)
            if record:
                soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct, tree_carbon2, oldtree_ct2, youngtree_ct2 = self.state 
                data["soil_carbons"].append(soil_carbon.item())
                data["tree_carbons"].append(tree_carbon.item())
                data["product_carbons"].append(product_carbon.item())
                data["oldtree_cts"].append(oldtree_ct.item())
                data["youngtree_cts"].append(youngtree_ct.item())
                data["tree_carbons2"].append(tree_carbon2.item())
                data["oldtree_cts2"].append(oldtree_ct2.item())
                data["youngtree_cts2"].append(youngtree_ct2.item())
        if record:
            timestamp = time.time()
            with open(os.path.join(outdirectory, f"{timestamp}_econrun.json"), 'w') as f:
                f.write(json.dumps(data, indent=4))
            plot_eco(data, outdirectory, f"{timestamp}_econrun")
            print(f"Saved: {timestamp}_econrun.json/jpg")
    
    
    def total_carbon(state):
        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct, tree_carbon2, oldtree_ct2, youngtree_ct2 = state
        return soil_carbon + tree_carbon + product_carbon + tree_carbon2


def eco_run(parameters, undisturbed_steps=1000, delta = 0.01, record=False):
    """Runs for undisturbed_steps*delta years. """
    with U.make_session(num_cpu=8):
        env = FCEnv(**parameters) # Create the environment
        env.reset()
        env.eco_step(undisturbed_steps, delta, record=record)

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions. Used as qfunc"""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def dqn2(parameters, dqnparams, max_episode_num = 10, steps_per_episode=100, save_frequency=100):
    """ main difference in how things are looped. This one has max_episode_num and steps_per_episode.
        returns 'act': ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
     """
    # https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py
    start_time = time.time()
    with U.make_session(num_cpu=32):
        env = FCEnv(**parameters) # Create the environment
        ## 1. Create all the functions/configurations necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name), # input placeholder for specific observation space
            q_func=model,
            num_actions=env.action_space.n,  # (1,)
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            gamma = dqnparams["gamma"] # relatively long-term
        )
        act_params = { 'make_obs_ph': lambda name: ObservationInput(env.observation_space, name=name),
            'q_func': model, 'num_actions': env.action_space.n }
        act = deepq.deepq.ActWrapper(act, act_params) # wrapper for saving later
        ## 2. Create buffer, exploration schedule, and reward weights
        replay_buffer = ReplayBuffer(50000) # Create the replay buffer, Max number of transitions to store in the buffer
        exploration_start = dqnparams["exploration_start"]
        exploration_end = dqnparams["exploration_end"]
        exploration_timestep = dqnparams["exploration_timestep"]
        exploration_change = (exploration_end-exploration_start)/exploration_timestep # linear schedule for exploration
        ## 3. Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()
        exploration = exploration_start
        ## 4. Start training. Each episode trains to done or for 100 steps (1 step = 1 year)
        total_stepct = 0
        episode_rewards = [] # each ele = reward from one complete run until done
        actions_taken1, actions_taken2 = [], []
        for episode in range(max_episode_num): # each episode: restart run, keep learning on old weights
            state = env.reset()  # returns state
            episode_rewards.append(0)
            for step in range(steps_per_episode): # each step = 1 year, each episode = steps_per_episode years
                total_stepct+=1
                exploration = max(exploration + exploration_change, exploration_end)
                env.eco_step(99, delta = 0.01)
                action = act(state[None], stochastic=False)[0] if np.random.uniform(0,1) > exploration else np.int64(env.action_space.sample()) # observation obj (axis added), stochastic boolean, update
                new_state, rew, done, info = env.step(action) # updates state
                actions_taken1.append(int(action.item()/20)*5) # for species 1
                actions_taken2.append((action.item()%20)*5) # for species 2
                episode_rewards[-1] += rew
                ## Store transition in the replay buffer.
                replay_buffer.add(state, action, rew, new_state, float(done))
                state = new_state
                if done: break # go to next episode
                # if total_stepct > 100 and np.mean(episode_rewards[-101:-1]) >= 200:
                    # print(f"---SOLVED: {np.mean(episode_rewards[-101:-1])}---"), env.render()
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32) # Sample a batch of experiences
                train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            update_target() # Update target network periodically.
            # logger.record_tabular("steps", total_stepct)
            # logger.record_tabular("episodes", len(episode_rewards))
            # logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            # logger.record_tabular("% time spent exploring", int(exploration * 100)) #int(100 * exploration.value(total_stepct)
            # logger.dump_tabular()
            print(f"\n### episode {episode} ### episode reward={episode_rewards[-1]} | Total carbon={FCEnv.total_carbon(env.state)}")
            if episode%save_frequency==0 or episode==max_episode_num-1:
                env.reset()
                validate_run(env, act, os.path.join(outdirectory, f"{start_time}_{episode}"), parameters, dqnparams, num_run_steps=100)
            
        total_data = {"parameters": str(parameters), "dqnparams": str(dqnparams), "actions_taken1": actions_taken1, "actions_taken2": actions_taken2, 
                    "combined_actions": list(zip(actions_taken1, actions_taken2)), "episode_rewards": episode_rewards}
            # episode_rewards = np.array(episode_rewards)/steps_per_episode # reward per step/year
        plot_rewardsactions(total_data, outdirectory, f"{start_time}_training") # plot episode rewards
        print(f"Saved: {start_time}_training.jpg")
        with open(os.path.join(outdirectory, f"{start_time}_{episode}_final.json"), 'w') as f:      
            f.write(json.dumps(total_data, indent=4))
        print("\n------------------------ %s seconds ------------------------" % (time.time() - start_time)) # round(time.time() - start_time, 2)
        print(start_time)
        print(f"actions_taken={total_data['combined_actions']}, Number of actions_taken={len(actions_taken1)}")
        print(f"episode_rewards={episode_rewards}")
        # print(parameters, "\n")
        # with open("log.txt", 'a') as file:
            # file.write(f"\n\n\n#####{endtime}#####\n")
            # file.writelines(str(parameters))
            # file.write("\n* actions_taken\n")
            # file.writelines(str(np.array(actions_taken[-300:])))
            # file.write("\n* episode_rewards\n") # reward per step/year
            # file.writelines(str(episode_rewards))
            # file.write("\n* ending state\n")
            # file.writelines(str(env.state))
        # return actions_taken, episode_rewards, env.state


def validate_run(env, act, filename, parameters, dqnparams, num_run_steps=100):
    """Each step corresopnds to 1 year"""
    # act = deepq.learn(env, network='mlp', total_timesteps=0, load_path=ckpt_path)
    data = {"parameters": str(parameters), "dqnparams": str(dqnparams), "step_rewards": [], "actions_taken1": [], "actions_taken2": [],
        "soil_carbons":[], "tree_carbons":[], "product_carbons":[], "oldtree_cts":[], "youngtree_cts":[],
        "tree_carbons2":[], "oldtree_cts2":[], "youngtree_cts2":[]}
    state = env.reset()  # returns state
    for step in range(num_run_steps):
        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct, tree_carbon2, oldtree_ct2, youngtree_ct2 = env.state 
        data["soil_carbons"].append(soil_carbon.item())
        data["tree_carbons"].append(tree_carbon.item())
        data["product_carbons"].append(product_carbon.item())
        data["oldtree_cts"].append(oldtree_ct.item())
        data["youngtree_cts"].append(youngtree_ct.item())
        data["tree_carbons2"].append(tree_carbon2.item())
        data["oldtree_cts2"].append(oldtree_ct2.item())
        data["youngtree_cts2"].append(youngtree_ct2.item())

        env.eco_step(99, delta = 0.01)
        action = act(state[None], stochastic=False)[0]
        new_state, rew, done, info = env.step(action) # updates state
        data["step_rewards"].append(rew) # reward per step/year
        data["actions_taken1"].append(int(action.item()/20)*5) # for species 1
        data["actions_taken2"].append((action.item()%20)*5) # for species 2
        state = new_state
        if done: 
            print("Done")
            return
    plot_with_rewards(data, "", filename)
    start_end = { "starting total cabron": FCEnv.total_carbon(env.initial_state).item(), "ending total cabron": FCEnv.total_carbon(env.state).item(),
        "ending state": env.state.tolist(),  "starting state": env.initial_state.tolist() }
    data["start_end"]=start_end
    with open(os.path.join(f"{filename}.json"), 'w') as f:      
        f.write(json.dumps(data, indent=4))
    act.save_act(f"{filename}.pkl")
    print(f"Saved: {filename}.jpg/json/pkl")



# def run_ckpt(ckpt_path, num_run_steps=100):
#     """Each step corresopnds to 1 year"""
#     act = deepq.deepq.ActWrapper.load_act(ckpt_path) # returns Actwrapper
#     filename = ckpt_path[:-4]
#     env = FCEnv(**parameters)
#     # act = deepq.learn(env, network='mlp', total_timesteps=0, load_path=ckpt_path)
#     actions_taken = []
#     step_rewards = []
#     data = {"checkpoint path": ckpt_path, "step_rewards": [], "actions_taken": [],
#         "soil_carbons":[], "tree_carbons":[], "product_carbons":[], "oldtree_cts":[], "youngtree_cts":[]}
#     state = env.reset()  # returns state
#     for step in range(num_run_steps):
#         soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = env.state 
#         data["soil_carbons"].append(soil_carbon.item())
#         data["tree_carbons"].append(tree_carbon.item())
#         data["product_carbons"].append(product_carbon.item())
#         data["oldtree_cts"].append(oldtree_ct.item())
#         data["youngtree_cts"].append(youngtree_ct.item())
#         env.eco_step(99, delta = 0.01)
#         action = act(state[None], stochastic=False)[0]
#         new_state, rew, done, info = env.step(action) # updates state
#         data["step_rewards"].append(rew) # reward per step/year
#         data["actions_taken"].append(action.item())
#         state = new_state
#         if done: 
#             print("Done")
#             return
#     timestamp = time.time()
#     plot_with_rewards(data, "", f"{filename}_run_{timestamp}")
#     start_end = {  "starting total cabron": FCEnv.total_carbon(env.initial_state).item(), "ending total cabron": FCEnv.total_carbon(env.state).item(),
#         "ending state": env.state.tolist(),  "starting state": env.initial_state.tolist() }
#     data["start_end"]=start_end
#     with open(os.path.join(f"{filename}_run_{timestamp}.json"), 'w') as f:      
#         f.write(json.dumps(data, indent=4))
#     print(f"Saved: {filename}_run_{timestamp}.json")
#     print(f"actions_taken={data['actions_taken']}, Number of actions_taken={len(actions_taken)}")
#     print(f"step_rewards={data['step_rewards']}")
#     print(f"start_end={start_end}")



def plot_json():
    for f in os.listdir(outdirectory): 
        if not f.endswith("json"): # 1645376609.1076465_last_episode_vals.json
            continue
        print("\n", f)
        data = json.load(open(os.path.join(outdirectory, f)))
        plot_with_rewards(data, outdirectory, f[:-5])


def plot_with_rewards(data, directory, filename):
    x = list(range(len(data["oldtree_cts"])))
    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(12, 16)) # sharex='all'
    fig.subplots_adjust(hspace=0.3)
    plt.suptitle("Multispecies Forest and Carbon State over 100 Years under Trained Agent's Forest Management", fontsize=16) 
    axs[0].set_title("Number of Trees over 100 Years from Forest Management Plan", fontsize=15)
    axs[0].plot(x, data["oldtree_cts"], label="Old Trees - Species 1")
    axs[0].plot(x, data["youngtree_cts"], label="Young Trees - Species 1")
    axs[0].plot(x, data["oldtree_cts2"], label="Old Trees - Species 2")
    axs[0].plot(x, data["youngtree_cts2"], label="Young Trees - Species 2")
    axs[0].set_xlabel("Time (years)", fontsize=14)
    axs[0].set_ylabel("Number of Trees", fontsize=14)
    axs[0].legend()
    
    axs[1].set_title("Amount of Carbon Sequestered over 100 Years from Multispecies Forest Management Plan", fontsize=15)
    axs[1].plot(x, data["soil_carbons"], label="Soil Carbon")
    axs[1].plot(x, data["tree_carbons"], label="Tree 1 Carbon")
    axs[1].plot(x, data["tree_carbons2"], label="Tree 2 Carbon")
    axs[1].plot(x, data["product_carbons"], label="Product Carbon")
    axs[1].set_xlabel("Time (years)", fontsize=14)
    axs[1].set_ylabel("Amount of Carbon Sequestered (Mg)", fontsize=14)
    axs[1].legend(fontsize=13)

    axs[2].set_title("Agent's Choice of Actions and Rewards over 100 Years for Multispecies Forest", fontsize=15)
    axs[2].set_xlabel("Time (years)", fontsize=14)
    axs[2].plot(x, data["step_rewards"], label="Rewards", color="blue")
    axs[2].set_ylabel("Rewards", fontsize=14)
    axtwin=axs[2].twinx()  # make a plot with different y-axis on same graph
    axtwin.plot(x, data["actions_taken1"], label="Percentage of Trees Harvested - Species 1", color="green")
    axtwin.plot(x, data["actions_taken2"], label="Percentage of Trees Harvested - Species 2", color="brown")
    axtwin.set_ylabel("Yearly Percentage of Trees Harvested",fontsize=14)
    axs[2].legend(loc='upper left', fontsize=13)
    axtwin.legend(loc='upper right', fontsize=13)

    plt.savefig(os.path.join(directory, f"{filename}.jpg"))
    plt.tight_layout(pad=2.0)


def plot_rewardsactions(data, directory, filename):
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12, 6)) # sharex='all'
    plt.suptitle("Rewards Agent Received over Training for Multispecies Forest", fontsize=16)
    axs.plot(list( range(1, len(data["episode_rewards"])+1) ), data["episode_rewards"])
    axs.set_xlabel("Training Episodes", fontsize=15)
    axs.set_ylabel("Total Rewards Per Episode ", fontsize=15)
    plt.savefig(os.path.join(directory, f"{filename}_rewards.jpg"))

    x = [e/100 for e in list( range(1, len(data["actions_taken1"])+1) )] # convert from steps to episode
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12, 6)) # sharex='all'
    plt.suptitle("Agent's Choice of Actions over Training for Multispecies Forest", fontsize=16) 
    axs.plot(x, data["actions_taken1"])
    axs.plot(x, data["actions_taken2"])
    axs.legend(fontsize=13)
    axs.set_xlabel("Training Episodes", fontsize=15)
    axs.set_ylabel("Agent's Choice of Actions over Training", fontsize=15)
    plt.savefig(os.path.join(directory, f"{filename}_actions.jpg"))


def plot_eco(data, directory, filename):
    """No rewards/action plot. Currenly only used by eco_run"""
    x = [e/100 for e in list(range(len(data["oldtree_cts"])))] # convert from steps to year
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(12, 12)) # sharex='all'
    plt.suptitle("Multispecices Forest and Carbon State over 100 Years under Trained Agent's Forest Management", fontsize=16) 
    axs[0].set_title("Number of Trees over 100 Years from Forest Management Plan", fontsize=15)
    axs[0].plot(x, data["oldtree_cts"], label="Old Trees - Species 1")
    axs[0].plot(x, data["youngtree_cts"], label="Young Trees - Species 1")
    axs[0].plot(x, data["oldtree_cts2"], label="Old Trees - Species 2")
    axs[0].plot(x, data["youngtree_cts2"], label="Young Trees - Species 2")
    axs[0].set_xlabel("Time (years)", fontsize=15)
    axs[0].set_ylabel("Number of Trees", fontsize=15)
    axs[0].legend(fontsize=14)
    axs[1].set_title("Amount of Carbon Sequestered over 100 Years from Forest Management Plan", fontsize=15)
    axs[1].plot(x, data["soil_carbons"], label="Soil Carbon")
    axs[1].plot(x, data["tree_carbons"], label="Tree 1 Carbon")
    axs[1].plot(x, data["tree_carbons2"], label="Tree 2 Carbon")
    axs[1].plot(x, data["product_carbons"], label="Product Carbon")
    axs[1].set_xlabel("Time (years)", fontsize=15)
    axs[1].set_ylabel("Amount of Carbon Sequestered (Mg)", fontsize=15)
    axs[1].legend(fontsize=14)
    plt.savefig(os.path.join(directory, f"{filename}.jpg"))

def compute_biodiversity(oldtree_ct, youngtree_ct, oldtree_ct2, youngtree_ct2):
    """Shannon's entropy. Biodiversity peaked when porp1=porp2=0.5"""
    total_ct = oldtree_ct+youngtree_ct+oldtree_ct2+youngtree_ct2
    if total_ct==0: return -5 # penalty for harvesting all trees 
    porp1 = (oldtree_ct+youngtree_ct)/total_ct
    porp2 = (oldtree_ct+youngtree_ct2)/total_ct
    bio = math.exp( -1*(porp1)*math.log(porp1) -1*(porp2)*math.log(porp2) )
    return bio

def tanh(x):
    return (math.exp(2*x)-1)/(math.exp(2*x)+1)

def param_search():
    # x = np.linspace(0, 1, nx)
    # y = np.linspace(0, 1, ny)
    # xv, yv = np.meshgrid(x, y)
    return



if __name__ == '__main__':
    parameters = {
        "initial_state": np.array([150, 76, 0, 328, 328, 76, 328, 328], dtype=np.float32), 
        # soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct, tree_carbon2, oldtree_ct2, youngtree_ct2
        "litterfall_rate": 0.01, # tree litterfall rate, K_T
        "soil_decay": 0.02, # soil decay, K_S
        "product_decay": 0.01, # product decay, K_P
        "tree_mature_rate": 1/37, # maturation rate (young to old), m_1
        "tree_mature_rate2": 1/25, # maturation rate (young to old), m_1                          ## NEW
        "tree_density" : 656, # tree density, d_t
        "tree_carbon_y_tonhayear": 3, # carbon tons/ha/year for young trees, NPP_y
        "tree_carbon_o_tonhayear": 1, # carbon tons/ha/year for old trees, NPP_o  
        "tree_carbon_y_tonhayear2": 5, # carbon tons/ha/year for young trees, NPP_y               ## NEW
        "tree_carbon_o_tonhayear2": 0.5, # carbon tons/ha/year for old trees, NPP_o               ## NEW
        "reproduction_o":0.6, # reproduction rate for old, r_o
        "reproduction_y":0.2, # reproduction rate for young, reproduction_y
        "reproduction_o2":0.8, # reproduction rate for old, r_o                                   ## NEW
        "reproduction_y2":0.01, # reproduction rate for young, reproduction_y                     ## NEW
        "tree_carbon_o_mgpertree": 0.076/8, # amount of carbon per old tree (Megagrams), C_o
        "tree_carbon_y_mgpertree": 0.038/8, # amount of carbon per young tree (Megagrams), C_y
        "tree_carbon_o_mgpertree2": 0.076/4, # amount of carbon per old tree (Megagrams), C_o     ## NEW
        "tree_carbon_y_mgpertree2": 0.038/8, # amount of carbon per young tree (Megagrams), C_y   ## NEW
        "tree_death_o": 0.05, # death rate of old tree, death_o
        "tree_death_y": 0.25, # death rate of young tree, death_y,
        "tree_death_o2": 0.05, # death rate of old tree, death_o                                  ## NEW
        "tree_death_y2": 0.25, # death rate of young tree, death_y,                               ## NEW
        "carrying_capacity": 10000 # carraying capacity of the forest
    }
    dqnparams = {
        "gamma": 0.99, # discount factor
        "exploration_start": 1,
        "exploration_end": 0.02,
        "exploration_timestep": 10000, # updated each step of each episode
        "carbon_reward_weight": 0,
        "biodiversiy_weight": 0.5,
        "abundance_weight": 0.5
    }
    outdirectory = "multi-bio+abundance"
    if not os.path.exists(outdirectory): os.makedirs(outdirectory)
    # eco_run(parameters, undisturbed_steps=10000, record=True) # 100 years
    dqn2(parameters, dqnparams, max_episode_num = 500, steps_per_episode=100, save_frequency=10)
    # run_ckpt("baseline/1645405750.802999_499.pkl", num_run_steps=100)
    # "output2/1645390930.6857603_last_episode_vals.pkl"
    print("\n", outdirectory, "\nDONE\n")

        
    
    


##### NOTES
# act: function that choses an action given an observation
    # _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
    #                          outputs=deterministic_actions,
    #                          givens={update_eps_ph: -1.0, stochastic_ph: True},
    #                          updates=[update_eps_expr])
    # q_values = q_func(observations_ph.get(), num_actions, scope="q_func")         
    # deterministic_actions = tf.argmax(q_values, axis=1)

    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/deepq/build_graph.py#L146
    # https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L134
    #  function(inputs, outputs, updates=None, givens=None):
    #     """Just like Theano function. Take a bunch of tensorflow placeholders and expressions
    #     computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    #     values to be fed to the input's placeholders and produces the values of the expressions
    #     in outputs.
    #     Input values can be passed in the same order as inputs or can be provided as kwargs based
    #     on placeholder name (passed to constructor or accessible via placeholder.op.name).
    #     Example:
    #         x = tf.placeholder(tf.int32, (), name="x")
    #         y = tf.placeholder(tf.int32, (), name="y")
    #         z = 3 * x + 2 * y
    #         lin = function([x, y], z, givens={y: 0})
    #         with single_threaded_session():
    #             initialize()
    #             assert lin(2) == 6
    #             assert lin(x=3) == 9
    #             assert lin(2, 2) == 10
    #             assert lin(x=2, y=3) == 12
# train: function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:
    # td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
    # loss = huber_loss[td_error]
# https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/deepq/build_graph.py#L317