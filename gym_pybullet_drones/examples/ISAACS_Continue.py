'''
Continuation Training Script for ISAACS_tauGDA models.

Loads a previously saved ISAACS_tauGDA checkpoint and resumes training, preserving
the replay buffer and timestep counter so WandB/TensorBoard logs continue smoothly.

Follows safeHoverTrainISAACS_tauGDA.py structure exactly — all env and algorithm
hyperparameters are identical; only --model_path is added to specify the checkpoint.

Note: leaderboard_eval_env cannot be pickled and is re-attached after loading.
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
DEFAULT_MODEL_PATH = "results/best_model.zip"

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACTION_STRING = 'rpm'
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_STEPS = 1000000
DEFAULT_GAMMA = .99
DEFAULT_EVAL_FREQ = 5000

# ISAACS / tau-GDA hyperparameters
DEFAULT_DISTURBANCE_ENT_COEF = 1.0
DEFAULT_ACTOR_UPDATE_INTERVAL = 1
DEFAULT_DISTURBANCE_BOUND = .2
DEFAULT_TAU_A = 1.0

# SafeHoverAviary initialization parameters
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

# Leaderboard defaults
DEFAULT_LEADERBOARD_UPDATE_FREQ = 0
DEFAULT_LEADERBOARD_K_U = 5
DEFAULT_LEADERBOARD_K_D = 5
DEFAULT_LEADERBOARD_N_EPS = 5


def run(
        multiagent=DEFAULT_MA,
        action_space=DEFAULT_ACTION_STRING,
        train_steps=DEFAULT_STEPS,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        plot=True,
        colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO,
        local=True,
        segment_path=DEFAULT_SEGMENT_PATH,
        num_segments=DEFAULT_NUM_SEGMENTS,
        random_init=DEFAULT_RANDOM_INIT,
        biased_random=DEFAULT_BIASED_RANDOM,
        bias_threshold=DEFAULT_BIASED_RANDOM_THRESHOLD,
        gamma=DEFAULT_GAMMA,
        disturbance_ent_coef=DEFAULT_DISTURBANCE_ENT_COEF,
        actor_update_interval=DEFAULT_ACTOR_UPDATE_INTERVAL,
        disturbance_bound=DEFAULT_DISTURBANCE_BOUND,
        tau_a=DEFAULT_TAU_A,
        eval_freq=DEFAULT_EVAL_FREQ,
        random_vel=DEFAULT_RANDOM_VEL,
        vel_range=DEFAULT_VEL_RANGE,
        ang_vel_range=DEFAULT_ANG_VEL_RANGE,
        hover_threshold=DEFAULT_HOVER_THRESHOLD,
        hover_steps=DEFAULT_HOVER_STEPS,
        episode_len_sec=DEFAULT_EPISODE_LEN_SEC,
        leaderboard_update_freq=DEFAULT_LEADERBOARD_UPDATE_FREQ,
        leaderboard_k_u=DEFAULT_LEADERBOARD_K_U,
        leaderboard_k_d=DEFAULT_LEADERBOARD_K_D,
        leaderboard_n_eps=DEFAULT_LEADERBOARD_N_EPS,
        model_path=DEFAULT_MODEL_PATH,
):
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    # Timestamped save directory for this continuation run
    filename = os.path.join(output_folder, 'save-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    act_space = ActionType(action_space)
    print(f"[INFO] Continuing training from: {model_path}")
    print(f"[INFO] random_init={random_init}, random_vel={random_vel}, vel_range={vel_range}")

    env_kwargs = dict(
        obs=DEFAULT_OBS,
        act=act_space,
        random_init=random_init,
        biased_random=biased_random,
        bias_threshold=bias_threshold,
        random_vel=random_vel,
        vel_range=vel_range,
        ang_vel_range=ang_vel_range,
        hover_threshold=hover_threshold,
        hover_steps=hover_steps,
        episode_len_sec=episode_len_sec,
    )

    train_env = make_vec_env(SafeHoverAviary, env_kwargs=env_kwargs, n_envs=4, seed=0)
    eval_env = SafeHoverAviary(**env_kwargs)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    disturbance_space = spaces.Box(
        low=np.array([-disturbance_bound] * 4, dtype=np.float32),
        high=np.array([disturbance_bound] * 4, dtype=np.float32),
    )

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": train_steps,
        "env_name": "SafeHoverAviary",
        "algorithm": "ISAACS_tauGDA_continue",
        "continued_from": model_path,
        "gamma": gamma,
        "disturbance_ent_coef": disturbance_ent_coef,
        "actor_update_interval": actor_update_interval,
        "tau_a": tau_a,
        "disturbance_bound": disturbance_bound,
        "leaderboard_update_freq": leaderboard_update_freq,
        "leaderboard_k_u": leaderboard_k_u,
        "leaderboard_k_d": leaderboard_k_d,
        "leaderboard_n_eps": leaderboard_n_eps,
    }

    wandb_run = wandb.init(
        project="SafeDroneFlight",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )

    # Load checkpoint and attach the new training environment.
    # leaderboard_eval_env is not picklable, so it's re-attached manually after loading.
    model = ISAACS_tauGDA.load(model_path, env=train_env)
    model.leaderboard_eval_env = eval_env
    model.leaderboard.eval_env = eval_env
    print(f"[INFO] Loaded model. Resuming from timestep {model.num_timesteps}.")

    target_reward = 400.0
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename + '/',
        log_path=filename + '/',
        eval_freq=int(eval_freq),
        deterministic=True,
        render=False,
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{wandb_run.id}",
        verbose=2,
    )

    # reset_num_timesteps=False keeps the existing timestep counter and replay buffer
    model.learn(
        total_timesteps=train_steps if local else int(1e2),
        callback=[eval_callback, wandb_callback],
        log_interval=100,
        reset_num_timesteps=False,
    )

    model.save(filename + '/final_model.zip')
    print(filename)

    with np.load(filename + '/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j]) + "," + str(data['results'][j][0]))

    if local:
        input("Press Enter to continue...")

    if os.path.isfile(filename + '/best_model.zip'):
        path = filename + '/best_model.zip'
    else:
        print("[ERROR]: no best model saved under", filename)
        return

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
    print(f"\n\n\nMean reward {mean_reward} +- {std_reward}\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward,
              "\tTerminated:", terminated, "\tTruncated:", truncated)
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
    parser = argparse.ArgumentParser(description='Continue training an ISAACS_tauGDA checkpoint')
    parser.add_argument('--model_path',          default=DEFAULT_MODEL_PATH,              type=str,       help='Path to the saved ISAACS_tauGDA checkpoint (.zip)', metavar='')
    parser.add_argument('--multiagent',          default=DEFAULT_MA,                      type=str2bool,  help='Whether to use multiagent mode (default: False)', metavar='')
    parser.add_argument('--gui',                 default=DEFAULT_GUI,                     type=str2bool,  help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',        default=DEFAULT_RECORD_VIDEO,            type=str2bool,  help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',       default=DEFAULT_OUTPUT_FOLDER,           type=str,       help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',               default=DEFAULT_COLAB,                   type=bool,      help='Whether example is being run by a notebook (default: False)', metavar='')
    parser.add_argument('--action_space',        default=DEFAULT_ACTION_STRING,           type=str,       help='Action space string: rpm, pid, vel, one_d_rpm, one_d_pid', metavar='')
    parser.add_argument('--train_steps',         default=DEFAULT_STEPS,                   type=int,       help='Additional timesteps to train', metavar='')
    parser.add_argument('--segment_path',        default=DEFAULT_SEGMENT_PATH,            type=str2bool,  help='Whether to segment the path to a random target', metavar='')
    parser.add_argument('--num_segments',        default=DEFAULT_NUM_SEGMENTS,            type=int,       help='How many path segments to use', metavar='')
    parser.add_argument('--random_init',         default=DEFAULT_RANDOM_INIT,             type=str2bool,  help='Whether to randomize starting positions each episode', metavar='')
    parser.add_argument('--biased_random',       default=DEFAULT_BIASED_RANDOM,           type=str2bool,  help='Whether to use biased random sampling during warmup', metavar='')
    parser.add_argument('--bias_threshold',      default=DEFAULT_BIASED_RANDOM_THRESHOLD, type=float,    help='Threshold for biased random flag', metavar='')
    parser.add_argument('--gamma',               default=DEFAULT_GAMMA,                   type=float,     help='Discount factor', metavar='')
    parser.add_argument('--disturbance_ent_coef', default=DEFAULT_DISTURBANCE_ENT_COEF,  type=float,     help='Entropy coefficient for the disturbance actor', metavar='')
    parser.add_argument('--actor_update_interval', default=DEFAULT_ACTOR_UPDATE_INTERVAL, type=int,      help='Gradient steps between control actor updates', metavar='')
    parser.add_argument('--disturbance_bound',   default=DEFAULT_DISTURBANCE_BOUND,       type=float,     help='Bounds for the disturbance space', metavar='')
    parser.add_argument('--tau_a',               default=DEFAULT_TAU_A,                   type=float,     help='Disturbance timescale ratio (LR_dist = tau_a * LR_actor)', metavar='')
    parser.add_argument('--eval_freq',           default=DEFAULT_EVAL_FREQ,               type=int,       help='How often to run eval', metavar='')
    parser.add_argument('--random_vel',          default=DEFAULT_RANDOM_VEL,              type=str2bool,  help='Randomize initial velocity at episode start', metavar='')
    parser.add_argument('--vel_range',           default=DEFAULT_VEL_RANGE,               type=float,     help='Linear velocity randomization range (m/s)', metavar='')
    parser.add_argument('--ang_vel_range',       default=DEFAULT_ANG_VEL_RANGE,           type=float,     help='Angular velocity randomization range (rad/s)', metavar='')
    parser.add_argument('--hover_threshold',     default=DEFAULT_HOVER_THRESHOLD,         type=float,     help='Safety margin above which hover counter increments', metavar='')
    parser.add_argument('--hover_steps',         default=DEFAULT_HOVER_STEPS,             type=int,       help='Consecutive steps above hover_threshold before truncating', metavar='')
    parser.add_argument('--episode_len_sec',     default=DEFAULT_EPISODE_LEN_SEC,         type=int,       help='Hard episode length cap in seconds', metavar='')
    parser.add_argument('--leaderboard_update_freq', default=DEFAULT_LEADERBOARD_UPDATE_FREQ, type=int,  help='Gradient steps between leaderboard tournament updates (0=disabled)', metavar='')
    parser.add_argument('--leaderboard_k_u',     default=DEFAULT_LEADERBOARD_K_U,         type=int,       help='Max control policies retained in leaderboard', metavar='')
    parser.add_argument('--leaderboard_k_d',     default=DEFAULT_LEADERBOARD_K_D,         type=int,       help='Max disturbance policies retained in leaderboard', metavar='')
    parser.add_argument('--leaderboard_n_eps',   default=DEFAULT_LEADERBOARD_N_EPS,       type=int,       help='Episodes per matchup in leaderboard tournament', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
