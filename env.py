from cgitb import reset
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
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule




import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"WARNING", category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging 
logging.getLogger('tensorflow').disabled = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # surpress warning
tf.get_logger().setLevel('ERROR')
TON2MG = 0.907185

class FCEnv(gym.Env):  # the forest carbon env
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_state=np.array([150, 76, 0, 328, 328], dtype=np.float32), 
                litterfall_rate=0.019, soil_decay=1/6.2, product_decay=1/100, tree_mature_rate=1/37, tree_density=656, tree_carbon_y_tonhayear=2.5, tree_carbon_o_tonhayear=0.2, 
                reproduction_o=0.6, reproduction_y=0.2, tree_carbon_o_mgpertree = 0.076, tree_carbon_y_mgpertree = 0.038, tree_death_o = 0.05, tree_death_y=0.1, carrying_capacity=10000):
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

        self.eco_step_matrix = np.array(
                [[-1*soil_decay, litterfall_rate, 0, tree_carbon_o_mgpertree*tree_death_o, tree_carbon_y_mgpertree*tree_death_y], # soil_carbon
                 [0, -1*litterfall_rate, 0, -1*tree_carbon_o_mgpertree*tree_death_o, -1*tree_carbon_y_mgpertree*tree_death_y],   # tree_carbon
                 [0, 0, -1*product_decay, 0, 0],   # product_carbon
                 [0, 0, 0, -1*tree_death_o, tree_mature_rate],    # oldtree_ct
                 [0, 0, 0, 0, -1*tree_mature_rate+-1*tree_death_y]] # youngtree_ct
            ) 
        eigenvalues, eigenvectors = np.linalg.eig(self.eco_step_matrix)
        print("eigenvalues:", eigenvalues)
        print("eigenvectors:", eigenvectors)
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
        tree_carbon = max(0, tree_carbon-num_tree_cut*self.tree_carbon_o_mgpertree) # each tree has 0.1 
        product_carbon += num_tree_cut*self.tree_carbon_o_mgpertree*3/4 # loses half
        # 3. Calculate rewards
        carbon_sequestered = tree_carbon+soil_carbon+product_carbon - orig_total_carbon
        # econ_profit = 
        # print(f"\taction={action}, frac_tree_cut={frac_tree_cut}, num_tree_cut={num_tree_cut}, carbon_sequestered={carbon_sequestered}")
        # TODO: each reward softmax -> weight

        reward = 1.0 * carbon_sequestered #can have different reward function

        done = bool(youngtree_ct + oldtree_ct == 0.0)
    
        # -> At t+1, we see state_t1+ carbon_update + action_update
        self.state = np.array([soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct], dtype=np.float32)
        return self.state, reward, done, {}

    def eco_step(self, num_steps = 1, delta = 0.01, record=False):
        """Updates self.state based on differential equation model of the ecosystem of forest and carbon """
        if record:
            data = {"soil_carbons":[], "tree_carbons":[], "product_carbons":[], "oldtree_cts":[], "youngtree_cts":[]}
        for i in range(num_steps):
            soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = self.state 
            f = TON2MG * (youngtree_ct/self.tree_density * self.tree_carbon_y_tonhayear + oldtree_ct/self.tree_density * self.tree_carbon_o_tonhayear) # tree growth's intake of carbon
            g = (self.reproduction_y*youngtree_ct+self.reproduction_o*oldtree_ct)*(1-(oldtree_ct+youngtree_ct)/self.carrying_capacity) # reproduction of trees
            # if (i % 100 == 0) or (i == num_steps-1):
            #     print(f"eco_step {i}: self.state={self.state}, f={f}, g={g}")
            self.state += delta * (np.matmul(self.eco_step_matrix, self.state) + np.array([0, f, 0, 0, g]))  # 2d and 1d  # (5,)
            if record:
                soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = self.state 
                data["soil_carbons"].append(soil_carbon.item())
                data["tree_carbons"].append(tree_carbon.item())
                data["product_carbons"].append(product_carbon.item())
                data["oldtree_cts"].append(oldtree_ct.item())
                data["youngtree_cts"].append(youngtree_ct.item())
        if record:
            timestamp = time.time()
            with open(os.path.join(outdirectory, f"{timestamp}_econrun.json"), 'w') as f:      
                f.write(json.dumps(data, indent=4))
            plot_eco(data, outdirectory, f"{timestamp}_econrun")
            print(f"Saved: {timestamp}_econrun.json/jpg")
    
    
    def total_carbon(state):
        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = state
        return soil_carbon + tree_carbon + product_carbon


def eco_run(parameters, undisturbed_steps=1000, delta = 0.01, record=False):
    """Runs for undisturbed_steps*delta years. """
    with U.make_session(num_cpu=8):
        env = FCEnv(**parameters) # Create the environment
        env.reset()
        env.eco_step(undisturbed_steps, delta, record=record)

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def dqn2(parameters, dqnparams, max_episode_num = 10, steps_per_episode=100, save_frequency=100):
    """main difference in how things are looped. This one has max_episode_num and steps_per_episode.
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
        ## 2. Create buffer and exploration schedule
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
        actions_taken = []
        for episode in range(max_episode_num): # each episode: restart run, keep learning on old weights
            state = env.reset()  # returns state
            episode_rewards.append(0)
            for step in range(steps_per_episode): # each step = 1 year, each episode = steps_per_episode years
                total_stepct+=1
                exploration = max(exploration + exploration_change, exploration_end)
                env.eco_step(99, delta = 0.01)
                action = act(state[None], stochastic=False)[0] if np.random.uniform(0,1) > exploration else np.int64(env.action_space.sample()) # observation obj (axis added), stochastic boolean, update
                new_state, rew, done, info = env.step(action) # updates state
                actions_taken.append(action.item())
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
            
        total_data = {"parameters": str(parameters), "dqnparams": str(dqnparams), "actions_taken": actions_taken, "episode_rewards": [e.item() for e in episode_rewards]}  # to include not just last 100
            # episode_rewards = np.array(episode_rewards)/steps_per_episode # reward per step/year
        plot_rewardsactions(total_data, outdirectory, f"{start_time}_training") # plot episode rewards
        print(f"Saved: {start_time}_totaltraining.jpg")
        with open(os.path.join(outdirectory, f"{start_time}_{episode}_final.json"), 'w') as f:      
            f.write(json.dumps(total_data, indent=4))
        print("\n------------------------ %s seconds ------------------------" % (time.time() - start_time)) # round(time.time() - start_time, 2)
        print(start_time)
        print(f"actions_taken={actions_taken}, Number of actions_taken={len(actions_taken)}")
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
    data = {"parameters": str(parameters), "dqnparams": str(dqnparams), "step_rewards": [], "actions_taken": [],
        "soil_carbons":[], "tree_carbons":[], "product_carbons":[], "oldtree_cts":[], "youngtree_cts":[]}
    state = env.reset()  # returns state
    for step in range(num_run_steps):
        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = env.state 
        data["soil_carbons"].append(soil_carbon.item())
        data["tree_carbons"].append(tree_carbon.item())
        data["product_carbons"].append(product_carbon.item())
        data["oldtree_cts"].append(oldtree_ct.item())
        data["youngtree_cts"].append(youngtree_ct.item())
        env.eco_step(99, delta = 0.01)
        action = act(state[None], stochastic=False)[0]
        new_state, rew, done, info = env.step(action) # updates state
        data["step_rewards"].append(rew.item()) # reward per step/year
        data["actions_taken"].append(action.item())
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



def run_ckpt(ckpt_path, num_run_steps=100):
    """Each step corresopnds to 1 year"""
    act = deepq.deepq.ActWrapper.load_act(ckpt_path) # returns Actwrapper
    filename = ckpt_path[:-4]
    env = FCEnv(**parameters)
    # act = deepq.learn(env, network='mlp', total_timesteps=0, load_path=ckpt_path)
    actions_taken = []
    step_rewards = []
    data = {"checkpoint path": ckpt_path, "step_rewards": [], "actions_taken": [],
        "soil_carbons":[], "tree_carbons":[], "product_carbons":[], "oldtree_cts":[], "youngtree_cts":[]}
    state = env.reset()  # returns state
    for step in range(num_run_steps):
        soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct = env.state 
        data["soil_carbons"].append(soil_carbon.item())
        data["tree_carbons"].append(tree_carbon.item())
        data["product_carbons"].append(product_carbon.item())
        data["oldtree_cts"].append(oldtree_ct.item())
        data["youngtree_cts"].append(youngtree_ct.item())
        env.eco_step(99, delta = 0.01)
        action = act(state[None], stochastic=False)[0]
        new_state, rew, done, info = env.step(action) # updates state
        data["step_rewards"].append(rew.item()) # reward per step/year
        data["actions_taken"].append(action.item())
        state = new_state
        if done: 
            print("Done")
            return
    timestamp = time.time()
    plot_with_rewards(data, "", f"{filename}_run_{timestamp}")
    start_end = {  "starting total cabron": FCEnv.total_carbon(env.initial_state).item(), "ending total cabron": FCEnv.total_carbon(env.state).item(),
        "ending state": env.state.tolist(),  "starting state": env.initial_state.tolist() }
    data["start_end"]=start_end
    with open(os.path.join(f"{filename}_run_{timestamp}.json"), 'w') as f:      
        f.write(json.dumps(data, indent=4))
    print(f"Saved: {filename}_run_{timestamp}.json")
    print(f"actions_taken={data['actions_taken']}, Number of actions_taken={len(actions_taken)}")
    print(f"step_rewards={data['step_rewards']}")
    print(f"start_end={start_end}")



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
    plt.suptitle("Forest and Carbon State over 100 Years under Trained Agent's Forest Management", fontsize=16) # timestamp
    axs[0].set_title("Number of Trees over 100 Years from Forest Management Plan", fontsize=15)
    axs[0].plot(x, data["oldtree_cts"], label="Old trees")
    axs[0].plot(x, data["youngtree_cts"], label="Young trees")
    axs[0].set_xlabel("Time (years)", fontsize=14)
    axs[0].set_ylabel("Number of Trees", fontsize=14)
    axs[0].legend()
    
    axs[1].set_title("Amount of Carbon Sequestered over 100 Years from Forest Management Plan", fontsize=15)
    axs[1].plot(x, data["soil_carbons"], label="Soil carbon")
    axs[1].plot(x, data["tree_carbons"], label="Tree carbon")
    axs[1].plot(x, data["product_carbons"], label="Product carbon")
    axs[1].set_xlabel("Time (years)", fontsize=14)
    axs[1].set_ylabel("Amount of Carbon Sequestered (Mg)", fontsize=14)
    axs[1].legend(fontsize=13)

    axs[2].set_title("Agent's Choice of Actions and Rewards over 100 Years", fontsize=15)
    axs[2].set_xlabel("Time (years)", fontsize=14)
    axs[2].plot(x, data["step_rewards"], label="Rewards", color="blue")
    axs[2].set_ylabel("Rewards", fontsize=14)
    axtwin=axs[2].twinx()  # make a plot with different y-axis on same graph
    axtwin.plot(x, data["actions_taken"], label="Percentage of Trees Harvested", color="green")
    axtwin.set_ylabel("Yearly Percentage of Trees Harvested",fontsize=14)
    axs[2].legend(loc='upper left', fontsize=13)
    axtwin.legend(loc='upper right', fontsize=13)

    plt.savefig(os.path.join(directory, f"{filename}.jpg"))
    plt.tight_layout(pad=2.0)


def plot_rewardsactions(data, directory, filename):
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 6)) # sharex='all'
    plt.suptitle("Rewards Agent Received over Training", fontsize=16) # timestamp
    axs.plot(list( range(1, len(data["episode_rewards"])+1) ), data["episode_rewards"])
    axs.set_xlabel("Training Episodes", fontsize=15)
    axs.set_ylabel("Total Rewards Per Episode ", fontsize=15)
    # axs[0].set_title("Number of Trees over 100 Years from Forest Management Plan", fontsize=13)
    plt.savefig(os.path.join(directory, f"{filename}_rewards.jpg"))

    x = [e/100 for e in list( range(1, len(data["actions_taken"])+1) )] # convert from steps to episode
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 6)) # sharex='all'
    plt.suptitle("Agent's Choice of Actions over Training", fontsize=16) # timestamp
    axs.plot(x, data["actions_taken"])
    axs.set_xlabel("Training Episodes", fontsize=15)
    axs.set_ylabel("Agent's Choice of Actions over Training", fontsize=15)
    # axs[0].set_title("Number of Trees over 100 Years from Forest Management Plan", fontsize=13)
    plt.savefig(os.path.join(directory, f"{filename}_actions.jpg"))


def plot_eco(data, directory, filename):
    """No rewards/action plot. Currenly only used by eco_run"""
    x = [e/100 for e in list(range(len(data["oldtree_cts"])))] # convert from steps to year
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(12, 12)) # sharex='all'
    plt.suptitle("Forest and Carbon State over 100 Years under Trained Agent's Forest Management", fontsize=16) # timestamp
    axs[0].set_title("Number of Trees over 100 Years from Forest Management Plan", fontsize=15)
    axs[0].plot(x, data["oldtree_cts"], label="Old trees")
    axs[0].plot(x, data["youngtree_cts"], label="Young trees")
    axs[0].set_xlabel("Time (years)", fontsize=15)
    axs[0].set_ylabel("Number of Trees", fontsize=15)
    axs[0].legend(fontsize=14)
    axs[1].set_title("Amount of Carbon Sequestered over 100 Years from Forest Management Plan", fontsize=15)
    axs[1].plot(x, data["soil_carbons"], label="soil_carbons")
    axs[1].plot(x, data["tree_carbons"], label="tree_carbons")
    axs[1].plot(x, data["product_carbons"], label="product_carbons")
    axs[1].set_xlabel("Time (years)", fontsize=15)
    axs[1].set_ylabel("Amount of Carbon Sequestered (Mg)", fontsize=15)
    axs[1].legend(fontsize=14)
    plt.savefig(os.path.join(directory, f"{filename}.jpg"))


def param_search():
    # x = np.linspace(0, 1, nx)
    # y = np.linspace(0, 1, ny)
    # xv, yv = np.meshgrid(x, y)
    return



if __name__ == '__main__':
    parameters = {
        "initial_state": np.array([150, 76, 0, 328, 328], dtype=np.float32), # soil_carbon, tree_carbon, product_carbon, oldtree_ct, youngtree_ct
        "litterfall_rate": 0.01, # tree litterfall rate, K_T
        "soil_decay": 0.02, # soil decay, K_S
        "product_decay": 0.01, # product decay, K_P
        "tree_mature_rate": 1/37, # maturation rate (young to old), m_1
        "tree_density" : 656, # tree density, d_t
        "tree_carbon_y_tonhayear": 1.5, # carbon released tons/ha/year for young trees, NPP_y
        "tree_carbon_o_tonhayear": 0.5, # carbon released tons/ha/year for old trees, NPP_o
        "reproduction_o":0.3, # reproduction rate for old, r_o
        "reproduction_y":0.1, # reproduction rate for young, reproduction_y
        "tree_carbon_o_mgpertree": 0.076/8, # amount of carbon per old tree (Megagrams), C_o
        "tree_carbon_y_mgpertree": 0.038/8, # amount of carbon per young tree (Megagrams), C_y
        "tree_death_o": 0.05, # death rate of old tree, death_o
        "tree_death_y": 0.25, # death rate of young tree, death_y,
        "carrying_capacity": 10000 # carraying capacity of the forest
    }
    dqnparams = {
        "gamma": 0.99, # discount factor
        "exploration_start": 1,
        "exploration_end": 0.02,
        "exploration_timestep": 10000 # updated each step of each episode
    }
    outdirectory = "baseline-moreH-carbonyear+reproduction"
    if not os.path.exists(outdirectory): os.makedirs(outdirectory)
    # eco_run(parameters, undisturbed_steps=10000, record=True) # 100 years
    dqn2(parameters, dqnparams, max_episode_num = 500, steps_per_episode=100, save_frequency=10)
    # run_ckpt("baseline/1645405750.802999_499.pkl", num_run_steps=100)
    # "output2/1645390930.6857603_last_episode_vals.pkl"
    print("\nDONE\n")

        
    
    


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