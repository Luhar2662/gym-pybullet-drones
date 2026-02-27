"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
trying to establish the control loop for moving drone to random place, so that it can be 
implemented with SafeHoverAviary (reset/housekeeping for episodes)

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.SafeHoverAviary import SafeHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.isaacs.safety_sac import SafetySAC
from gym_pybullet_drones.utils.utils import sync, str2bool

import torch as th


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1 #SINGLEAGENT
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_SEGMENT_PATH = True
DEFAULT_NUM_SEGMENTS = 1

DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_RANDOM_INIT = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 1
DEFAULT_MA = False
DEFAULT_GUARD_DURATION = 10
DEFAULT_SAFETY_THRESHOLD = .25

def segment(init_xyzs, goal_pos, segments) -> np.ndarray:
    target_waypoints = np.zeros((segments, 3))
    tot_offset = goal_pos - init_xyzs

    #split the offset into <segments> amount of offsets
    seg_offset = tot_offset / segments

    #generate the path 
    target_waypoints[0] = init_xyzs
    for i in range(segments):
        target_waypoints[i] = target_waypoints[i-1] + seg_offset
    
    return np.array(target_waypoints)

def target_generate(num_drones = 1) -> np.ndarray:
    return  np.array([[np.random.uniform(-1,1),
                               np.random.uniform(-1,1),
                               np.random.uniform(.1, 2)] for i in range(num_drones)])

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        segment_path = DEFAULT_SEGMENT_PATH,
        num_segments = DEFAULT_NUM_SEGMENTS,
        model_path = DEFAULT_MODEL_PATH,
        random_init = DEFAULT_RANDOM_INIT,
        multiagent = DEFAULT_MA,
        safety_threshold = DEFAULT_SAFETY_THRESHOLD,
        guard_duration = DEFAULT_GUARD_DURATION,
        ):
    #### Initialize the simulation #############################
    
    R = .3
    INIT_XYZS = np.array([[0,0,.1] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    
    #### Initialize and choose target pose ######################
    target = np.array([[np.random.uniform(-1,1),
                               np.random.uniform(-1,1),
                               np.random.uniform(.1, 2)] for i in range(num_drones)])
    
    print("randomly chose target: ", target)

    waypoints = np.array([segment(INIT_XYZS[i], target[i], num_segments) for i in range(num_drones)])
    print("waypoints: ", waypoints.shape)
    waypoint_ct = 0
    waypoint_dur = duration_sec / num_segments

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
        start_rpy = np.array([[0,
                            0,
                            0,
                            ] for i in range(1)])
    else:
        start_pos = np.array([[0,
                                0,
                                .1] for i in range(1)])
        start_rpy = np.array([[0,
                            0,
                            0,
                            ] for i in range(1)])
    INIT_XYZS = start_pos
    INIT_RPYS = start_rpy

    #### Create test environment ####
    if not multiagent:
        env = SafeHoverAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=start_pos, initial_rpys= start_rpy)
    else:
        env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder="logs_playback/",
                    colab=False)



    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])


    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    '''
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    '''

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    start = time.time()
    obs, _ = env.reset(seed=42, options={})

    guard_active = False
    guard_timesteps = guard_duration
    guard_triggered = 0

    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        #Update step:
        obs, reward, terminated, truncated, info = env.step(action)
        
        #update waypoint counter if needed
        if i != 0 and (i % (waypoint_dur*env.CTRL_FREQ)==0) and segment_path:
            waypoint_ct = waypoint_ct + 1
        
        ctrl_obs = env._computeObsForControl()
        #print("Observation given back: ", obs)

        #### Compute control for the current way point #############
        
        for j in range(num_drones):
            if segment_path:
                target_wp = waypoints[j, waypoint_ct, :]
            else:
                target_wp = np.hstack([target[j,0:2], target[j, 2]])
            print("target shape: ", np.hstack([target[j,0:2], target[j, 2]]))

            #for i in range(num_drones):
                #print("waypoints: ", segment(INIT_XYZS[i], target[0], num_segments))
                #current waypoint: waypoints[waypoint_ct, j, 0:2]
            

            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=ctrl_obs[j],
                                                                    target_pos=target_wp,
                                                                    #target_pos=np.hstack([target[j,0:2], target[j, 2]]),
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )
        #now, action holds the recommended action!
        fallback, _ = model.predict(obs, deterministic = True)

        obs2 = obs.squeeze()
        act2 = action.squeeze()

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
        q_pred = critic_check(obs_t, act_t)
        print("qnet average predicts: ", q_pred, ", for obs: ", obs2[0:3], ", act: ", act2)


        #CHECK SAFETY PER THRESHOLD
        # Substitution step for switching filter ######
        pred_avg = (q_pred[0].float() + q_pred[1].float()) / 2

        '''
        if pred_avg <= safety_threshold and not guard_active:
            #switching filter active!!

            guard_active = True
            action = fallback
            guard_triggered += 1
        
        elif guard_active: #REPLACEMENT
            action = fallback
            guard_timesteps -= 1

            #check duration
            if guard_timesteps <= 0:
                guard_active = False
                guard_timesteps = guard_duration

        else:
            pass #allow pid action through!
        '''
        

        

        
        #### Log the simulation ####################################
        print("target: ", target)
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([target[j, 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, start, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()
    print("########################## GUARD CALLED: ", guard_triggered, " times! #####################")

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--segment_path', default=DEFAULT_SEGMENT_PATH, type=bool,           help='Whether we segment the path to a random target (needed for bigger boxes)', metavar='')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--random_init', default=DEFAULT_RANDOM_INIT, type=str2bool,           help='whether to random init during rollout', metavar='')
    parser.add_argument('--num_segments', default=DEFAULT_NUM_SEGMENTS, type=int,           help='How many segments to make', metavar='')
    parser.add_argument('--guard_duration', default=DEFAULT_GUARD_DURATION, type=int,           help='How long the switching filter holds the replacement action', metavar='')
    parser.add_argument('--safety_threshold', default=DEFAULT_SAFETY_THRESHOLD, type=float,           help='q-value threshold for assessing switching filter activation', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
