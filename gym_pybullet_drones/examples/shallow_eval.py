import os
import time
import argparse
import numpy as np
import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.SafeHoverAviary import SafeHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.isaacs.safety_sac import SafetySAC
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_RANDOM_INIT = True
DEFAULT_NUM_RUNS = 5
DEFAULT_RANDOM_EVAL = True
DEFAULT_SAFE_INIT = False
DEFAULT_GUI = False
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 1
DEFAULT_MA = False

def eval_random_target():

        '''
        Pull from a specific safety margin with certainty, such that we always initialize from meaningful
        positions for testing!
        '''
        #print("called eval random")
        intervals = [[-.9,-.5],[.5,.9]]
        z_intervals = [[.1,.5],[1.5,1.9]]

        x_int = np.random.choice([0,1])
        y_int = np.random.choice([0,1])
        z_int = np.random.choice([0,1])

        random_target = np.array([[np.random.uniform(intervals[x_int][0],intervals[x_int][1]),
                               np.random.uniform(intervals[y_int][0],intervals[y_int][1]),
                               np.random.uniform(z_intervals[z_int][0],z_intervals[z_int][1])] for i in range(1)])

        random_target_rpy = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(1)])
        
        return random_target, random_target_rpy

def safe_random_target(model):
    verified = False
    
    while not verified:
        candidate_pos, candidate_rpy = eval_random_target()

        #check viability using a helper env
        test_env = SafeHoverAviary(gui=False, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=candidate_pos, initial_rpys= candidate_rpy)

        obs, _ = test_env.reset(seed=42, options={})
        action, _ = model.predict(obs, deterministic=True)

        obs_t, _ = model.policy.obs_to_tensor(obs)
        act_t = th.as_tensor(action, device=model.device).float()

        with th.no_grad():
            q1, q2 = model.policy.critic(obs_t, act_t)

        q_avg = np.mean([q1.item(), q2.item()])

        if q_avg > 0:
            verified = True
        else: 
            print("unsafe config: ", candidate_pos, ", resampling!")
    
    #verified safe, so pass on!

    return candidate_pos, candidate_rpy



    

def random_target():
    '''
    Randomly samples uniformly from the entire space
    '''
    random_target = np.array([[np.random.uniform(-1,1),
                                np.random.uniform(-1,1),
                                np.random.uniform(.1, 2)] for i in range(1)])
    random_target_rpy = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(1)])
    
    return random_target, random_target_rpy

def play(model_path=DEFAULT_MODEL_PATH, multiagent=DEFAULT_MA, gui=DEFAULT_GUI, random_init = DEFAULT_RANDOM_INIT, random_eval = DEFAULT_RANDOM_EVAL, safe_init = DEFAULT_SAFE_INIT, record = True, num_runs = DEFAULT_NUM_RUNS):
   
    #### Load saved model ####
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    model = SafetySAC.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    critic1 = model.policy.critic.qf0
    critic2 = model.policy.critic.qf1

    def critic_check(state, action):
        with th.no_grad():
            q1, q2 = model.policy.critic(state, action)
        return q1, q2

    
    if random_init:
        if random_eval:
            start_pos, start_rpy = eval_random_target()
        else:
            start_pos, start_rpy = random_target()
    else:
        start_pos = np.array([[0,
                                0,
                                .1] for i in range(1)])
        start_rpy = np.array([[0,
                            0,
                            0,
                            ] for i in range(1)])
    
        
    if random_eval:
        print("In evaluation mode for random generation!")

    

    #### Create test environment ####
    if not multiagent:
        env = SafeHoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=start_pos, initial_rpys= start_rpy, random_eval=random_eval)
    else:
        env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder="logs_playback/",
                    colab=False)


    unsafe_samples = 0
    unsafe_list = []

    for _ in range(num_runs):
        #### Run the simulation ####
        obs, _ = env.reset(seed=42, options={})
        action, _ = model.predict(obs, deterministic=True)
        
        obs_t, _ = model.policy.obs_to_tensor(obs)
        act_t = th.as_tensor(action, device=model.device).float()

        with th.no_grad():
            q1, q2 = model.policy.critic(obs_t, act_t)

        q_avg = np.mean([q1.item(), q2.item()])

        if q_avg < 0:
            unsafe_samples += 1
            unsafe_list.append([env.INIT_XYZS,env.INIT_RPYS])
        

        #RESET LOOP!
        if random_init:
            if random_eval:
                start_pos, start_rpy = eval_random_target()
            else:
                start_pos, start_rpy = random_target()
        else:
            start_pos = np.array([[0,
                                0,
                                .1] for i in range(1)])
            start_rpy = np.array([[0,
                            0,
                            0,
                            ] for i in range(1)])
        env.INIT_XYZS = start_pos
        env.INIT_RPYS = start_rpy
    env.close()

    success_rate = float(num_runs - unsafe_samples) / float(num_runs)
    print("Rate of success: ", success_rate)
    #logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--multiagent', type=bool, default=DEFAULT_MA, help='Whether to use MultiHoverAviary')
    parser.add_argument('--gui', type=bool, default=DEFAULT_GUI, help='Enable GUI rendering')
    parser.add_argument('--random_init', default=DEFAULT_RANDOM_INIT, type=str2bool,           help='whether to random init during rollout', metavar='')
    parser.add_argument('--random_eval', default=DEFAULT_RANDOM_EVAL, type=str2bool,           help='whether to randomize according to the eval bounds set in eval_random_target()', metavar='')
    
    parser.add_argument('--num_runs', default=DEFAULT_NUM_RUNS, type=int,           help='how many times to random init and play out sim', metavar='')
    
    args = parser.parse_args()

    play(**vars(args))
