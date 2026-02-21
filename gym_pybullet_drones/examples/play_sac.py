import os
import time
import argparse
import numpy as np
import gymnasium as gym
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

    critic1 = model.policy.qf1
    critic2 = model.policy.qf2

    def critic_check(state, action):
        return (critic1(state, action) + critic2(state, action)) * .5

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

    #### Create test environment ####
    if not multiagent:
        env = SafeHoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=start_pos, initial_rpys= start_rpy)
    else:
        env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder="logs_playback/",
                    colab=False)

    #### Run the simulation ####
    obs, _ = env.reset(seed=42, options={})
    start = time.time()

    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
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
        print("qnet average predicts: ", critic_check(obs, action), ", for obs: ", obs2, ", act: ", act2)

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break

    env.close()
    logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--multiagent', type=bool, default=DEFAULT_MA, help='Whether to use MultiHoverAviary')
    parser.add_argument('--gui', type=bool, default=DEFAULT_GUI, help='Enable GUI rendering')
    parser.add_argument('--random_init', default=DEFAULT_RANDOM_INIT, type=str2bool,           help='whether to random init during rollout', metavar='')
    args = parser.parse_args()

    play(**vars(args))
