"""Script demonstrating PPO training with an ISAACS_tauGDA safety filter.

Mirrors learnFiltered.py, but uses ISAACSFilteredHoverAviary so that a
pre-trained ISAACS_tauGDA fallback policy intercepts unsafe actions at each step.

Example
-------
In a terminal, run as:

    $ python ISAACSlearnFiltered.py --fallback_model_path results/best_model.zip
    $ python ISAACSlearnFiltered.py --fallback_model_path results/best_model.zip --fallback_threshold 0.2

Notes
-----
This is identical to learnFiltered.py except:
  1. ISAACSFilteredHoverAviary is used in place of SafetySACFilteredHoverAviary.
  2. ISAACS_tauGDA is loaded as the fallback model instead of SafetySAC.
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from wandb.integration.sb3 import WandbCallback
import wandb

from gym_pybullet_drones.isaacs.ISAACS_tauGDA import ISAACS_tauGDA
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.ISAACSFilteredHoverAviary import ISAACSFilteredHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_MA = False
DEFAULT_FALLBACK_MODEL_PATH = 'results/best_model.zip'
DEFAULT_FALLBACK_THRESHOLD = 0.2
DEFAULT_TERMINATE_ON_BOUNDARY = True
DEFAULT_TRAIN_STEPS = 350000
DEFAULT_EVAL_FREQ = 5000
DEFAULT_HORIZON = 0
DEFAULT_CHECK_Q = True
DEFAULT_CHECK_BOUNDS = True


def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI,
        plot=True,
        colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO,
        local=True,
        fallback_model_path=DEFAULT_FALLBACK_MODEL_PATH,
        fallback_threshold=DEFAULT_FALLBACK_THRESHOLD,
        terminate_on_boundary=DEFAULT_TERMINATE_ON_BOUNDARY,
        train_steps=DEFAULT_TRAIN_STEPS,
        eval_freq=DEFAULT_EVAL_FREQ,
        horizon=DEFAULT_HORIZON,
        check_q=DEFAULT_CHECK_Q,
        check_bounds=DEFAULT_CHECK_BOUNDS,
        ):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Load the ISAACS_tauGDA fallback filter ############################
    if not os.path.isfile(fallback_model_path):
        print(f"[ERROR] Fallback model not found: {fallback_model_path}")
        return
    fallback_model = ISAACS_tauGDA.load(fallback_model_path, device='cpu')
    fallback_model.policy.set_training_mode(False)
    print(f"[INFO] Loaded ISAACS_tauGDA fallback model from {fallback_model_path}")

    #### Create training and eval environments ##############################
    env_kwargs = dict(obs=DEFAULT_OBS,
                      act=DEFAULT_ACT,
                      fallback_model=fallback_model,
                      fallback_threshold=fallback_threshold,
                      terminate_on_boundary=terminate_on_boundary,
                      horizon=horizon,
                      check_q=check_q,
                      check_bounds=check_bounds)

    train_env = make_vec_env(ISAACSFilteredHoverAviary,
                             env_kwargs=env_kwargs,
                             n_envs=1,
                             seed=0
                             )
    eval_env = ISAACSFilteredHoverAviary(**env_kwargs)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### WandB config and init #############################################
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": train_steps,
        "env_name": "ISAACSFilteredHoverAviary",
        "algorithm": "PPO+ISAACS_tauGDA_filter",
        "fallback_model_path": fallback_model_path,
        "fallback_threshold": fallback_threshold,
        "terminate_on_boundary": terminate_on_boundary,
        "horizon": horizon,
        "check_q": check_q,
        "check_bounds": check_bounds,
    }
    wandb_run = wandb.init(
        project="SafeDroneFlight",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )

    #### Train PPO on the filtered environment #############################
    model = PPO('MlpPolicy',
                train_env,
                tensorboard_log=f"runs/{wandb_run.id}",
                verbose=1,
                device='cpu')

    target_reward = 467.
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
        verbose=2,
    )

    model.learn(total_timesteps=int(train_steps) if local else int(1e2),
                callback=[eval_callback, wandb_callback],
                log_interval=100)

    #### Save the model ####################################################
    model.save(filename+'/final_model.zip')
    print(filename)

    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ########################################################################

    if local:
        for env in train_env.envs:
            env.unwrapped.print_violations()
        input("Press Enter to continue...")

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
        return
    model = PPO.load(path, device='cpu')

    #### Show (and optionally record) the model's performance ##############
    test_env = ISAACSFilteredHoverAviary(**{**env_kwargs, 'gui': gui, 'record': record_video})
    test_env_nogui = ISAACSFilteredHoverAviary(**env_kwargs)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab)

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward,
              "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                       timestamp=i/test_env.CTRL_FREQ,
                       state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                       control=np.zeros(12))
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO training with ISAACS_tauGDA safety filter on ISAACSFilteredHoverAviary')
    parser.add_argument('--gui',                   default=DEFAULT_GUI,                  type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',          default=DEFAULT_RECORD_VIDEO,         type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',         default=DEFAULT_OUTPUT_FOLDER,        type=str,      help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',                 default=DEFAULT_COLAB,                type=bool,     help='Whether example is being run by a notebook (default: False)', metavar='')
    parser.add_argument('--fallback_model_path',   default=DEFAULT_FALLBACK_MODEL_PATH,  type=str,      help='Path to pre-trained ISAACS_tauGDA zip file used as safety filter', metavar='')
    parser.add_argument('--fallback_threshold',    default=DEFAULT_FALLBACK_THRESHOLD,   type=float,    help='Q-value threshold below which the filter activates', metavar='')
    parser.add_argument('--terminate_on_boundary', default=DEFAULT_TERMINATE_ON_BOUNDARY, type=str2bool, help='Terminate episode when drone leaves safe boundary (default: True)', metavar='')
    parser.add_argument('--train_steps',           default=DEFAULT_TRAIN_STEPS,          type=int,      help='Total PPO training timesteps (default: 350000)', metavar='')
    parser.add_argument('--eval_freq',             default=DEFAULT_EVAL_FREQ,            type=int,      help='Eval callback frequency in steps (default: 5000)', metavar='')
    parser.add_argument('--horizon',               default=DEFAULT_HORIZON,              type=int,      help='Predictive rollout horizon (0=disabled, uses one-step Q-check)', metavar='')
    parser.add_argument('--check_q',               default=DEFAULT_CHECK_Q,              type=str2bool, help='Check ISAACS Q-value at each lookahead step (default: True)', metavar='')
    parser.add_argument('--check_bounds',          default=DEFAULT_CHECK_BOUNDS,         type=str2bool, help='Check safety boundary at each lookahead step (default: True)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
