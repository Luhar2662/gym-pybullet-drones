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
DEFAULT_RANDOM_EVAL = False
DEFAULT_SAFE_INIT = False
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 1
DEFAULT_MA = False

def eval_random_target():

        '''
        Pull from a specific safety margin with certainty, such that we always initialize from meaningful
        positions for testing!
        '''
        print("called eval random")
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
        elif safe_init:
            start_pos, start_rpy = safe_random_target(model)
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
    time.sleep(3)
    for _ in range(num_runs):
        #### Run the simulation ####
        obs, _ = env.reset(seed=42, options={})
        
        print("random start point: ", obs)
        start = time.time()

        for i in range((env.EPISODE_LEN_SEC-1)*env.CTRL_FREQ):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            obs2 = obs.squeeze()
            act2 = action.squeeze()

            if DEFAULT_OBS == ObservationType.KIN:
                if not multiagent:
                    logger.log(drone=0,
                        timestamp=i/env.CTRL_FREQ,
                        state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2]),
                        control=np.zeros(12))
                else:
                    for d in range(DEFAULT_AGENTS):
                        logger.log(drone=d,
                            timestamp=i/env.CTRL_FREQ,
                            state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]]),
                            control=np.zeros(12))
                        
            #Track Safety Violations Here!
            ground_margin = obs2[2]
            left_margin = 1 - obs2[1]
            right_margin = obs2[1] + 1
            front_margin = 1 - obs2[0]
            back_margin = obs2[0] + 1
            ceil_margin = 2 - obs2[2]
            safety_margin = min(ground_margin, ceil_margin, back_margin, front_margin, left_margin, right_margin)
            if safety_margin <= 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Safety violation: state is [",obs2[0],", ", obs2[1],", ", obs2[2],"]")
                print("pausing for 2 seconds")
                time.sleep(2)
            obs_t, _ = model.policy.obs_to_tensor(obs)
            act_t = th.as_tensor(action, device=model.device).float()
            print("qnet average predicts: ", critic_check(obs_t, act_t), ", for obs: ", obs2[0:3], ", act: ", act2)

            env.render()
            sync(i, start, env.CTRL_TIMESTEP)
            if terminated:
                break

        #RESET LOOP!
        if random_init:
            if random_eval:
                start_pos, start_rpy = eval_random_target()
            elif safe_init:
                start_pos, start_rpy = safe_random_target(model)
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
    #logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--multiagent', type=bool, default=DEFAULT_MA, help='Whether to use MultiHoverAviary')
    parser.add_argument('--gui', type=bool, default=DEFAULT_GUI, help='Enable GUI rendering')
    parser.add_argument('--random_init', default=DEFAULT_RANDOM_INIT, type=str2bool,           help='whether to random init during rollout', metavar='')
    parser.add_argument('--random_eval', default=DEFAULT_RANDOM_EVAL, type=str2bool,           help='whether to randomize according to the eval bounds set in eval_random_target()', metavar='')
    parser.add_argument('--safe_init', default=DEFAULT_SAFE_INIT, type=str2bool,           help='whether to randomize according to a safety check', metavar='')
    parser.add_argument('--num_runs', default=DEFAULT_NUM_RUNS, type=int,           help='how many times to random init and play out sim', metavar='')
    
    args = parser.parse_args()

    play(**vars(args))
