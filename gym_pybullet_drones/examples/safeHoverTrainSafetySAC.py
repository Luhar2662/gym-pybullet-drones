"""Main Training Script for models using Safety SAC

Class SafeHoverAviary is used as learning env for the SAC SB3 algorithm.

Adapted from the standard 'learn.py' training script from gym_pybullet_drones

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch


from stable_baselines3 import SAC
from gym_pybullet_drones.isaacs.safety_sac import SafetySAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from wandb.integration.sb3 import WandbCallback
import wandb

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.SafeHoverAviary import SafeHoverAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType



class ObsSqueezeWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.squeeze(obs, axis=0), info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        return np.squeeze(obs, axis=0), reward, done, trunc, info

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACTION_STRING = 'rpm'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False
DEFAULT_STEPS = 1000000
DEFAULT_GAMMA = .99

# Default arguments for random initializing behavior defined in SafeHoverAviary
DEFAULT_SEGMENT_PATH = True
DEFAULT_NUM_SEGMENTS = 1
DEFAULT_RANDOM_INIT = False
DEFAULT_BIASED_RANDOM = True
DEFAULT_BIASED_RANDOM_THRESHOLD = .5

def run(multiagent=DEFAULT_MA, action_space = DEFAULT_ACTION_STRING, train_steps=DEFAULT_STEPS, output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, segment_path = DEFAULT_SEGMENT_PATH, num_segments = DEFAULT_NUM_SEGMENTS,
        random_init = DEFAULT_RANDOM_INIT, biased_random=DEFAULT_BIASED_RANDOM, bias_threshold=DEFAULT_BIASED_RANDOM_THRESHOLD, gamma = DEFAULT_GAMMA):
    
    '''
    action_space: one of 'rpm', 'pid', 'vel', 'one_d_rpm', 'one_d_pid'
    train_steps: amount of timesteps to train the model
    random_init: whether to randomly initialize across the space at the start of each episode
    biased_random: whether to initialize using the "biased" initialization scheme (see SafeHoverAviary)
    bias_threshold: the percent chance of randomly initializing instead of drawing from bias intervals when biased_random is enabled
    '''
    
    # Establish timestamped save directory for model logging, so that subsequent runs do not overwrite
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    
    # Create Training and Eval environments
    

    act_space = ActionType(action_space)
    print("creating safehoverenv with random_init value: ", random_init)

    
    train_env = make_vec_env(SafeHoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=act_space, random_init=random_init, biased_random=biased_random, bias_threshold=bias_threshold),
                                 n_envs=4, # Parallel Environments as supported by PyBullet for more efficient training
                                 seed=0
                                 )
    
    #Eval env using biased random, but has used uniform random in the past. This should better preserve the "best" model for filtering purposes
    eval_env = SafeHoverAviary(obs=DEFAULT_OBS, act=act_space, random_init = random_init) 
    
    
    
    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    config = {
        "policy_type": "MlpPolicy", 
        "total_timesteps": 10000000,
        "env_name": "SafeHoverAviary"
    }

    # Log run to WandB
    wandb_run = wandb.init(
        project="SafeDroneFlight",
        config = config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )

    model = SafetySAC('MlpPolicy',
                train_env,
                tensorboard_log=f"runs/{wandb_run.id}",
                gamma = gamma,
                verbose=1)

    #### Callbacks for evaluation and wandb ##################
    target_reward = 400.0
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb_run.id}",
        verbose = 2,
    )


    model.learn(total_timesteps=train_steps if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=[eval_callback, wandb_callback],
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    #### Visualize Trained Model (Best) ########################
    if local:

        input("Press Enter to continue...")

    
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = SafetySAC.load(path)

    #### Show (and record a video of) the model's performance ##

    test_env = SafeHoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=act_space,
                               record=record_video)
    test_env_nogui = SafeHoverAviary(obs=DEFAULT_OBS, act=act_space)
    
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )
    
    test_env_nogui = ObsSqueezeWrapper(test_env_nogui)

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    # Control loop using trained fallback policy
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--action_space', default=DEFAULT_ACTION_STRING, type=str, help='string corresponding to the action space used for training', metavar='')
    parser.add_argument('--train_steps', default=DEFAULT_STEPS, type=int, help='Amount of timesteps to train if target isnt reached', metavar='')
    parser.add_argument('--segment_path', default=DEFAULT_SEGMENT_PATH, type=str2bool,           help='Whether we segment the path to a random target (needed for bigger boxes)', metavar='')
    parser.add_argument('--num_segments', default=DEFAULT_NUM_SEGMENTS, type=int,           help='How many segments to make', metavar='')
    parser.add_argument('--random_init', default=DEFAULT_RANDOM_INIT, type=str2bool,           help='whether to call warmup() and randomize starting positions for policy training', metavar='')
    parser.add_argument('--biased_random', default=DEFAULT_BIASED_RANDOM, type=str2bool,           help='when calling warmup(), whether to initialize uniformly across full space, or to use biased random sampling', metavar='')
    parser.add_argument('--bias_threshold', default=DEFAULT_BIASED_RANDOM_THRESHOLD, type=float,           help='threshold for biased random flag', metavar='')
    parser.add_argument('--gamma', default='DEFAULT_GAMMA', type=float,           help='Default discount factor', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
