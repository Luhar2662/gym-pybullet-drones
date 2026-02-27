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
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 1
DEFAULT_MA = False

def play(model_path=DEFAULT_MODEL_PATH, multiagent=DEFAULT_MA, gui=DEFAULT_GUI, random_init = DEFAULT_RANDOM_INIT):
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
        start_pos = np.array([[np.random.uniform(-1,1),
                                np.random.uniform(-1,1),
                                np.random.uniform(.1, 2)] for i in range(1)])
        start_rpy = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(1)])
    else:
        start_pos = np.array([[0,
                                0,
                                .1] for i in range(1)])
        start_rpy = np.array([[0,
                            0,
                            0,
                            ] for i in range(1)])

        
        obs_t, _ = model.policy.obs_to_tensor(obs)
        act_t = th.as_tensor(action, device=model.device).float()
        print("qnet average predicts: ", critic_check(obs_t, act_t), ", for obs: ", obs2, ", act: ", act2)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--multiagent', type=bool, default=DEFAULT_MA, help='Whether to use MultiHoverAviary')
    parser.add_argument('--gui', type=bool, default=DEFAULT_GUI, help='Enable GUI rendering')
    parser.add_argument('--random_init', default=DEFAULT_RANDOM_INIT, type=str2bool,           help='whether to random init during rollout', metavar='')
    args = parser.parse_args()

    play(**vars(args))
