'''
Main Training Script for models using ISAACS_tauGDA (tau-GDA variant of ISAACS)

Uses SafeHoverAviary as the learning environment, and uses the ISAACS_tauGDA.py and isaacs_utils.py
implementation for the algorithm. ISAACS_tauGDA extends ISAACS with sequential tau-GDA actor updates
from MAGICS (arXiv:2409.13867) for improved minimax convergence.

Adapted from safeHoverTrainISAACS.py
'''

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from gym_pybullet_drones.isaacs.ISAACS_tauGDA import ISAACS_tauGDA
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
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_STEPS = 1000000
DEFAULT_GAMMA = .99
DEFAULT_EVAL_FREQ = 5000

#Disturbance specific defaults for ISAACS
DEFAULT_DISTURBANCE_ENT_COEF = 1.0
DEFAULT_ACTOR_UPDATE_INTERVAL = 1
DEFAULT_DISTURBANCE_BOUND = .2

# tau-GDA specific default
DEFAULT_TAU_A = 1.0

# Default arguments for random initializing behavior defined in SafeHoverAviary
DEFAULT_SEGMENT_PATH = True
DEFAULT_NUM_SEGMENTS = 1
DEFAULT_RANDOM_INIT = False
DEFAULT_BIASED_RANDOM = True
DEFAULT_BIASED_RANDOM_THRESHOLD = .5
DEFAULT_RANDOM_VEL = False
DEFAULT_VEL_RANGE = 1.0
DEFAULT_ANG_VEL_RANGE = 1.0
DEFAULT_HOVER_THRESHOLD = .8
DEFAULT_HOVER_STEPS = 30
DEFAULT_EPISODE_LEN_SEC = 6

# Leaderboard / tournament defaults
DEFAULT_LEADERBOARD_UPDATE_FREQ = 0   # 0 = disabled
DEFAULT_LEADERBOARD_K_U = 5
DEFAULT_LEADERBOARD_K_D = 5
DEFAULT_LEADERBOARD_N_EPS = 5

def run(multiagent=DEFAULT_MA,
        action_space = DEFAULT_ACTION_STRING,
        train_steps=DEFAULT_STEPS,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        plot=True,
        colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO,
        local=True,
        segment_path = DEFAULT_SEGMENT_PATH,
        num_segments = DEFAULT_NUM_SEGMENTS,
        random_init = DEFAULT_RANDOM_INIT,
        biased_random=DEFAULT_BIASED_RANDOM,
        bias_threshold=DEFAULT_BIASED_RANDOM_THRESHOLD,
        gamma = DEFAULT_GAMMA,
        disturbance_ent_coef = DEFAULT_DISTURBANCE_ENT_COEF,
        actor_update_interval = DEFAULT_ACTOR_UPDATE_INTERVAL,
        disturbance_bound = DEFAULT_DISTURBANCE_BOUND,
        tau_a = DEFAULT_TAU_A,
        eval_freq = DEFAULT_EVAL_FREQ,
        random_vel = DEFAULT_RANDOM_VEL,
        vel_range = DEFAULT_VEL_RANGE,
        ang_vel_range = DEFAULT_ANG_VEL_RANGE,
        hover_threshold = DEFAULT_HOVER_THRESHOLD,
        hover_steps = DEFAULT_HOVER_STEPS,
        episode_len_sec = DEFAULT_EPISODE_LEN_SEC,
        leaderboard_update_freq = DEFAULT_LEADERBOARD_UPDATE_FREQ,
        leaderboard_k_u = DEFAULT_LEADERBOARD_K_U,
        leaderboard_k_d = DEFAULT_LEADERBOARD_K_D,
        leaderboard_n_eps = DEFAULT_LEADERBOARD_N_EPS,
):
    '''
    action_space: one of 'rpm', 'pid', 'vel', 'one_d_rpm', 'one_d_pid'
    train_steps: amount of timesteps to train the model
    random_init: whether to randomly initialize across the space at the start of each episode
    biased_random: whether to initialize using the "biased" initialization scheme (see SafeHoverAviary)
    bias_threshold: the percent chance of randomly initializing instead of drawing from bias intervals when biased_random is enabled
    gamma: discount factor for training
    disturbance_ent_coef: entropy coefficient for the disturbance actor to effect exploration
    actor_update_interval: how many gradient steps to wait before updating actor (by default, 1 = every step)
    tau_a: disturbance timescale ratio (effective LR_dist = tau_a * LR_actor); >= 1.0
    '''

    # Establish timestamped save directory for model logging, so that subsequent runs do not overwrite
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # Create Training and Eval environments

    act_space = ActionType(action_space)
    print("creating safehoverenv with random_init value: ", random_init)


    train_env = make_vec_env(SafeHoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=act_space, random_init=random_init, biased_random=biased_random, bias_threshold=bias_threshold,
                                                 random_vel=random_vel, vel_range=vel_range, ang_vel_range=ang_vel_range,
                                                 hover_threshold=hover_threshold, hover_steps=hover_steps, episode_len_sec=episode_len_sec),
                                 n_envs=4, # Parallel Environments as supported by PyBullet for more efficient training
                                 seed=0
                                 )

    #Eval env using biased random, but has used uniform random in the past. This should better preserve the "best" model for filtering purposes
    eval_env = SafeHoverAviary(obs=DEFAULT_OBS, act=act_space, random_init=random_init, biased_random=biased_random, bias_threshold=bias_threshold, random_vel=random_vel, vel_range=vel_range, ang_vel_range=ang_vel_range,
                                                 hover_threshold=hover_threshold, hover_steps=hover_steps, episode_len_sec=episode_len_sec)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #Define ISAACS disturbance space (defaults to being an additive noise model)
    disturbance_space = spaces.Box(
        low=np.array([-disturbance_bound]*4, dtype=np.float32),
        high=np.array([disturbance_bound]*4, dtype=np.float32),
    )

    #### Create the model #######################################
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 10000000,
        "env_name": "SafeHoverAviary",
        "algorithm": "ISAACS_tauGDA",
        "gamma": gamma,
        "disturbance_ent_coef": disturbance_ent_coef,
        "actor_update_interval": actor_update_interval,
        "tau_a": tau_a,
        "leaderboard_update_freq": leaderboard_update_freq,
        "leaderboard_k_u": leaderboard_k_u,
        "leaderboard_k_d": leaderboard_k_d,
        "leaderboard_n_eps": leaderboard_n_eps,
    }

     # Log run to WandB
    wandb_run = wandb.init(
        project="SafeDroneFlight",
        config = config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )

    #Create ISAACS_tauGDA policy model
    model = ISAACS_tauGDA(
        'MlpPolicy',
        train_env,
        gradient_steps=-1,
        tensorboard_log=f"runs/{wandb_run.id}",
        gamma=gamma,
        disturbance_space=disturbance_space,
        disturbance_ent_coef=disturbance_ent_coef,
        actor_update_interval=actor_update_interval,
        tau_a=tau_a,
        leaderboard_eval_env=eval_env,
        leaderboard_update_freq=leaderboard_update_freq,
        leaderboard_k_u=leaderboard_k_u,
        leaderboard_k_d=leaderboard_k_d,
        leaderboard_n_eps=leaderboard_n_eps,
        verbose=1,
    )

    #### Callbacks for evaluation and wandb ##################
    target_reward = 400.0
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(eval_freq),
                                 deterministic=True,
                                 render=False)
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb_run.id}",
        verbose = 2,
    )


    #### Train the model #######################################
    model.learn(total_timesteps=train_steps if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=[eval_callback, wandb_callback],
                log_interval=100)

    model.save(filename+'/final_model.zip')
    print(filename)

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
    model = ISAACS_tauGDA.load(path)

    test_env = SafeHoverAviary(gui=gui, obs=DEFAULT_OBS, act=act_space, record=record_video)
    test_env_nogui = SafeHoverAviary(obs=DEFAULT_OBS, act=act_space)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=DEFAULT_AGENTS if multiagent else 1,
        output_folder=output_folder,
        colab=colab,
    )

    test_env_nogui = ObsSqueezeWrapper(test_env_nogui)

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward", mean_reward, "+-", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(
                    drone=0,
                    timestamp=i / test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                    control=np.zeros(12),
                )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(
                        drone=d,
                        timestamp=i / test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                        control=np.zeros(12),
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
    parser = argparse.ArgumentParser(description='ISAACS_tauGDA training script for SafeHoverAviary')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,                     type=str2bool,  help='Whether to use multiagent mode (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,                    type=str2bool,  help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,           type=str2bool,  help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,          type=str,       help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,                  type=bool,      help='Whether example is being run by a notebook (default: False)', metavar='')
    parser.add_argument('--action_space',       default=DEFAULT_ACTION_STRING,          type=str,       help='Action space string: rpm, pid, vel, one_d_rpm, one_d_pid', metavar='')
    parser.add_argument('--train_steps',        default=DEFAULT_STEPS,                  type=int,       help='Amount of timesteps to train', metavar='')
    parser.add_argument('--segment_path',       default=DEFAULT_SEGMENT_PATH,           type=str2bool,  help='Whether to segment the path to a random target', metavar='')
    parser.add_argument('--num_segments',       default=DEFAULT_NUM_SEGMENTS,           type=int,       help='How many path segments to use', metavar='')
    parser.add_argument('--random_init',        default=DEFAULT_RANDOM_INIT,            type=str2bool,  help='Whether to randomize starting positions each episode', metavar='')
    parser.add_argument('--biased_random',      default=DEFAULT_BIASED_RANDOM,          type=str2bool,  help='Whether to use biased random sampling during warmup', metavar='')
    parser.add_argument('--bias_threshold',     default=DEFAULT_BIASED_RANDOM_THRESHOLD, type=float,   help='Threshold for biased random flag', metavar='')
    parser.add_argument('--gamma',              default=DEFAULT_GAMMA,                  type=float,     help='Discount factor', metavar='')
    parser.add_argument('--disturbance_ent_coef', default=DEFAULT_DISTURBANCE_ENT_COEF, type=float,    help='Entropy coefficient for the disturbance actor', metavar='')
    parser.add_argument('--actor_update_interval', default=DEFAULT_ACTOR_UPDATE_INTERVAL, type=int,    help='Gradient steps between control actor updates', metavar='')
    parser.add_argument('--disturbance_bound', default=DEFAULT_DISTURBANCE_BOUND, type=float,   help='bounds for the disturbance space', metavar='')
    parser.add_argument('--tau_a',             default=DEFAULT_TAU_A,             type=float,   help='Disturbance timescale ratio (LR_dist = tau_a * LR_actor)', metavar='')
    parser.add_argument('--eval_freq',         default=DEFAULT_EVAL_FREQ,         type=int,     help='how often to run eval', metavar='')
    parser.add_argument('--random_vel',      default=DEFAULT_RANDOM_VEL,    type=str2bool, help='Randomize initial velocity at episode start', metavar='')
    parser.add_argument('--vel_range',       default=DEFAULT_VEL_RANGE,     type=float,    help='Linear velocity randomization range (m/s)', metavar='')
    parser.add_argument('--ang_vel_range',   default=DEFAULT_ANG_VEL_RANGE, type=float,    help='Angular velocity randomization range (rad/s)', metavar='')
    parser.add_argument('--hover_threshold', default=DEFAULT_HOVER_THRESHOLD,  type=float, help='Safety margin above which hover counter increments', metavar='')
    parser.add_argument('--hover_steps',     default=DEFAULT_HOVER_STEPS,   type=int,   help='Consecutive steps above hover_threshold before truncating', metavar='')
    parser.add_argument('--episode_len_sec', default=DEFAULT_EPISODE_LEN_SEC,     type=int,   help='Hard episode length cap in seconds', metavar='')
    parser.add_argument('--leaderboard_update_freq', default=DEFAULT_LEADERBOARD_UPDATE_FREQ, type=int, help='Gradient steps between leaderboard tournament updates (0=disabled)', metavar='')
    parser.add_argument('--leaderboard_k_u', default=DEFAULT_LEADERBOARD_K_U, type=int, help='Max control policies retained in leaderboard', metavar='')
    parser.add_argument('--leaderboard_k_d', default=DEFAULT_LEADERBOARD_K_D, type=int, help='Max disturbance policies retained in leaderboard', metavar='')
    parser.add_argument('--leaderboard_n_eps', default=DEFAULT_LEADERBOARD_N_EPS, type=int, help='Episodes per matchup in leaderboard tournament', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
