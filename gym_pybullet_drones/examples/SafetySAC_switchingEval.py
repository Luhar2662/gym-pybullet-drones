'''
Safety Filter Evaluation script: Candidate policy determined by PID control in CtrlAviary (to take advantage of
pre-tuned controllers), and filter is implemented using a SafetySAC critic.

Filter style: simple switching. When q_val prediction falls below a given threshold, the fallback actor 
is activated (guard mode) and is prioritized for a set guad duration before returning control to PID.

SafeHoverAviary observationsa re reconstructed from CtrlAviary state to give valid observations to the critic.

'''

import os
import time
import argparse
from collections import deque
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.SafeHoverAviary import SafeHoverAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.isaacs.safety_sac import SafetySAC

import torch as th

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_DURATION_SEC = 6
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_NUM_SEGMENTS = 1
DEFAULT_SEGMENT_PATH = True
DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_RANDOM_INIT = False
DEFAULT_GUARD_DURATION = 10
DEFAULT_SAFETY_THRESHOLD = 0.25
DEFAULT_SAFE_TARGET = False
DEFAULT_BIAS_TARGET = False

#Match ctrl_freq from the SafeHoverAviary instance that training occured in for an action buffer
SAFE_CTRL_FREQ = 30
ACTION_BUFFER_SIZE = SAFE_CTRL_FREQ // 2

####### HELPER FUNCTIONS ##################

def segment(init_xyzs, goal_pos, segments):
    '''
    Helper method that segments the target pose into a set of waypoints for more granular control and 
    PID acceleration
    '''
    target_waypoints = np.zeros((segments, 3))
    tot_offset = goal_pos - init_xyzs
    seg_offset = tot_offset / segments
    target_waypoints[0] = init_xyzs
    for i in range(segments):
        target_waypoints[i] = target_waypoints[i - 1] + seg_offset
    return target_waypoints

def create_action_buffer():
        '''
        helper function to generate an action buffer with no actions preloaded (all zero rpms)
        '''
        actions = [np.zeros((1,4), dtype = np.float32) for _ in range(ACTION_BUFFER_SIZE)]
        return deque(actions, maxlen=ACTION_BUFFER_SIZE)

def build_safe_obs(state, act_buffer):
    '''
    Helper function to create a SafeHoverAviary observation from state vector and action buffer.

    SafeHoverAviary observations: pos [0:3], rpy [7:10], vel [10:13], ang_vel [13:16].
    Then, append actions from buffer

    returns: obs --> array of observation size
    '''
    #pull necessary info from the state as outlined in CtrlAviary (skip quaternions)
    base_state = np.hstack([state[0:3], #xyz
                            state[7:10], #rpy
                            state[10:13], #v_x, v_y, v_z
                            state[13:16] #ang vels
                            ]) 
    obs = base_state.reshape(1,12).astype(np.float32) #conform to expected safe obs

    for action in act_buffer:
        obs = np.hstack([obs, action]) #stack actions from buffer on top to build full observation
    
    return obs

def normalized_to_rpm(action, hover):
    '''
    Helper function that processes action (see _preprocessAction() in BaseRLAviary)
    '''
    return hover * (1 + .05 * action)


def rpm_to_normalized(rpm, hover):
    '''
    helper function that essentially 'unprocesses' actions which are processed by BaseRLAviary's
    _preprocessAction() handler (as above)
    '''
    return np.clip((rpm / hover - 1.0) / .05, -1, 1).astype(np.float32)

def add_bounds_box(p_client, x_lim=(-1, 1), y_lim=(-1, 1), z_lim=(0, 2), color=[0, 0.5, 1]):
    '''
    Draw the 12 edges of the safety bounding box using PyBullet debug lines.
    '''
    x0, x1 = x_lim
    y0, y1 = y_lim
    z0, z1 = z_lim
    corners = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bottom face
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # top face
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # verticals
    ]
    for a, b in edges:
        p.addUserDebugLine(corners[a], corners[b], lineColorRGB=color, lineWidth=1.5,
                           physicsClientId=p_client)

def add_target_marker(pos, p_client):
    '''
    Helper function that appends the target pos to the visualization using the pybullet client
    that is active for the simulation. Represent target pos as a red sphere!
    '''

    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=.05, rgbaColor=[1,0,0,.5], physicsClientId = p_client)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere, basePosition=pos, physicsClientId=p_client)
    p.addUserDebugText( #print position of target following pyb examples
        f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})", textPosition=pos + np.array([0, 0, 0.1]), textColorRGB=[1, 0, 0],
        textSize=1.2, physicsClientId=p_client,
    )

def random_target():
    target = np.array([[np.random.uniform(-.75, .75),
                        np.random.uniform(-.75, .75),
                        np.random.uniform(0.2, 2.0)]])
    return target

def biased_target():

        '''
        Pull from a specific safety margin with certainty, such that we always initialize from meaningful
        positions for testing!
        '''
        
        intervals = [[-.9,-.5],[.5,.9]]
        z_intervals = [[.1,.5],[1.5,1.9]]

        x_int = np.random.choice([0,1])
        y_int = np.random.choice([0,1])
        z_int = np.random.choice([0,1])

        random_target = np.array([[np.random.uniform(intervals[x_int][0],intervals[x_int][1]),
                               np.random.uniform(intervals[y_int][0],intervals[y_int][1]),
                               np.random.uniform(z_intervals[z_int][0],z_intervals[z_int][1])] for i in range(1)])

        random_target_rpy = np.array([[0,0,0
                            ] for i in range(1)])
        
        return random_target, random_target_rpy

def safe_random_init(model, threshold):
        verified = False
        while not verified:
                candidate_pos, candidate_rpy = biased_target()
                

                #check viability using a helper env
                test_env = SafeHoverAviary(gui=False, initial_xyzs=candidate_pos, initial_rpys= candidate_rpy)

                obs, _ = test_env.reset(seed=42, options={})
                action, _ = model.predict(obs, deterministic=True)

                obs_t, _ = model.policy.obs_to_tensor(obs)
                act_t = th.as_tensor(action, device=model.device).float()

                with th.no_grad():
                    q1, q2 = model.policy.critic(obs_t, act_t)

                q_avg = np.mean([q1.item(), q2.item()])

                if q_avg > threshold:
                    verified = True
                else: 
                    print("unsafe config: ", candidate_pos, ", resampling!")
    
            #verified safe, so pass on!
        return candidate_pos, candidate_rpy

######## MAIN LOOP ##############

def run( #follows PID control examples in gym_pybullet_drones
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        segment_path=DEFAULT_SEGMENT_PATH,
        num_segments=DEFAULT_NUM_SEGMENTS,
        model_path=DEFAULT_MODEL_PATH,
        random_init=DEFAULT_RANDOM_INIT,
        safety_threshold=DEFAULT_SAFETY_THRESHOLD,
        guard_duration=DEFAULT_GUARD_DURATION,
        safe_target = DEFAULT_SAFE_TARGET,
        bias_target = DEFAULT_BIAS_TARGET,
):
    # Load Safety Filter fallback model and critic from --model_path
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return
    model = SafetySAC.load(model_path)
    print(f"[INFO] Loaded SafetySAC model from {model_path}")

    #choose random target and start position
    if random_init:
        start_pos = np.array([[np.random.uniform(-1, 1),
                               np.random.uniform(-1, 1),
                               np.random.uniform(0.1, 2.0)]])
    else:
        start_pos = np.array([[0., 0., 0.1]])

    start_rpy = np.zeros((1, 3))

    if safe_target:
        safe_pos,_ = safe_random_init(model, safety_threshold)
        print("safe_pos: ", safe_pos)
        target = safe_pos
    elif bias_target:
        bias_pos,_ = biased_target()
        target = bias_pos[0]
    else:
        target = random_target()

    print(f"[INFO] Start: {start_pos[0]},  Target: {target[0]}")

    waypoints = np.array([segment(start_pos[i], target[i], num_segments) for i in range(num_drones)])
    waypoint_ct = 0
    waypoint_dur = duration_sec / num_segments

    #Create CtrlAviary to run experiment

    env = CtrlAviary(
        drone_model = drone,
        num_drones = num_drones,
        initial_xyzs = start_pos,
        initial_rpys = start_rpy,
        physics = physics,
        pyb_freq = simulation_freq_hz,
        ctrl_freq = SAFE_CTRL_FREQ, #must match SafeHoverAviary settings so that trained fallback is accurate
        gui = gui,
        user_debug_gui = user_debug_gui,
    )

    HOVER_RPM = env.HOVER_RPM #pull for use with normalization and processing of actions

    # create controller for PID actions
    ctrl = [DSLPIDControl(drone_model = drone) for _ in range(num_drones)] #should only be 1 drone for now

    logger = Logger(
        logging_freq_hz=int(env.CTRL_FREQ),
        num_drones=num_drones,
        output_folder="logs_pid_eval/",
        colab=False,
    )

    # Safety Filter setup and monitoring (mostly just tracking guard activation)
    guard_active = False
    guard_timesteps = guard_duration
    guard_activated = 0
    safety_violations = 0
    action_buffer = create_action_buffer() #create buffer to begin querying fallback model

    # Main Control Loop!
    action = np.zeros((num_drones, 4))
    env.reset(seed=42, options={})

    add_target_marker(target[0], env.getPyBulletClient()) #add target visualization
    add_bounds_box(env.getPyBulletClient()) #visualize safety bounds
    start = time.time()

    for i in range(int(duration_sec * env.CTRL_FREQ)):
        env.step(action) #step the environment at the start of each timestep
        state = env._getDroneStateVector(0) #pull state from environment

        drone_x, drone_y, drone_z = state[0], state[1], state[2] #pull xyz

        #Check if drone has exited bounds (not the same as safety), and terminate episode if so
        irrecoverable = abs(drone_x) > 1.5 or abs(drone_y) > 1.5 or drone_z > 2.5 or drone_z < 0
        if irrecoverable:
            print("out of bounds at step ", i, " at position:(", drone_x, ", ", drone_y, ", ", drone_z,")")
            env.reset(seed=42, options = {})
            add_target_marker(target[0], env.getPyBulletClient()) #marker goes away on reset()
            add_bounds_box(env.getPyBulletClient())
            
            #Reset relevant elements
            for c in ctrl: #reset pid controllers
                c.reset()

            action_buffer = create_action_buffer()
            guard_active = False
            guard_timesteps = guard_duration
            waypoint_ct = 0
            state = env._getDroneStateVector(0)
            drone_x, drone_y, drone_z = state[0], state[1], state[2] #pull xyz
        
        #Update waypoint counter if needed
        if i != 0 and segment_path and (i % int(waypoint_dur * env.CTRL_FREQ) == 0):
            waypoint_ct = min(waypoint_ct + 1, num_segments - 1)
        
        #PID Computes action
        candidate = np.zeros((num_drones,4))
        for j in range(num_drones):
            target_wp = waypoints[j, waypoint_ct] if segment_path else target[j]
            candidate[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP, state=state,
                                                                    target_pos=target_wp, target_rpy = start_rpy[j])
        
        #Filter candidate action by generating safe obs and querying
        safe_obs = build_safe_obs(state, action_buffer)
        norm_candidate = rpm_to_normalized(candidate, HOVER_RPM)

        obs_t, _ = model.policy.obs_to_tensor(safe_obs)
        act_t = th.as_tensor(norm_candidate, device=model.device).float()

        with th.no_grad():
            q1, q2 = model.policy.critic(obs_t, act_t)
        q_mean = np.mean([q1.item(), q2.item()])

        if q_mean < safety_threshold and not guard_active: #first violation, not in guard mode
            guard_active = True
            guard_timesteps = guard_duration
            guard_activated += 1

            print("Guard Triggered at step ", i, " at position:(", drone_x, ", ", drone_y, ", ", drone_z,")")
        
        if guard_active: #we are actively filtering!
            norm_fallback, _ = model.predict(safe_obs, deterministic = True)
            fallback = normalized_to_rpm(norm_fallback, HOVER_RPM) #raw RPM for CtrlAviary
            action = fallback
            norm_output = norm_fallback

            guard_timesteps -= 1

            if guard_timesteps <= 0: #switch away from guard
                guard_active = False
                guard_timesteps = guard_duration
        
        else: #guard inactive, pass through pid generated candidate
            action = candidate
            norm_output = norm_candidate
        
        action_buffer.append(norm_output.reshape(1,4).astype(np.float32))

        #Check for safety violations using margin function
        margin = min(drone_z, 2 - drone_z, 1 - drone_y, 1 + drone_y, 1 - drone_x, 1 + drone_x) #see SafeHoverAviary reward definition
        if margin <= 0:
            safety_violations += 1
            print("Safety Violation at step ", i, " at position:(", drone_x, ", ", drone_y, ", ", drone_z,")")

        #Check if we reached the vicinity of the target
        if abs(drone_x - target[0][0]) < .1 and abs(drone_y - target[0][1]) < .1 and abs(drone_z - target[0][2]) < .1:
            reached = True

        
        #Log iteration
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=np.hstack([state[0:3], np.zeros(4), state[7:10],
                              state[10:13], state[13:16], norm_output.flatten()]),
            control=np.hstack([target[0], start_rpy[0], np.zeros(6)]),
        )

        env.render()
        if gui:
            sync(i, start, env.CTRL_TIMESTEP)
    
    #iterations have finished!
    env.close()
    print("[RESULTS] Guard triggered:", guard_activated, " times!")
    print("[RESULTS] Safety violations:", safety_violations, " times!")
    print("[RESULTS] Reached target:", reached)

    logger.save()
    logger.save_as_csv("pid_eval")
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PID safety filter evaluation using CtrlAviary + SafetySAC')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,           type=DroneModel, help='drone model to be used', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,       type=int,        help='number of drones', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,          type=Physics,    help='PyBullet physics being used', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,              type=str2bool,   help='whether to use gui or not', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,    type=str2bool,   help='whether to record video', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,             type=str2bool,   help='whether to display RPM plot after completion', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,   type=str2bool,   help='show user debug gui', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,      help='what value to use for simulation frequency in hz', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,     type=int,        help='duration of episode', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,    type=str,        help='output folder', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,            type=bool,       metavar='')
    parser.add_argument('--segment_path',       default=DEFAULT_SEGMENT_PATH,     type=str2bool,   help='Whether to use the segment waypoint approach', metavar='')
    parser.add_argument('--num_segments',       default=DEFAULT_NUM_SEGMENTS,     type=int,        help='Number of segments that segment waypoint approach uses', metavar='')
    parser.add_argument('--model_path',         default=DEFAULT_MODEL_PATH,       type=str, help='model path for fallback policy used in filtering')
    parser.add_argument('--random_init',        default=DEFAULT_RANDOM_INIT,      type=str2bool,   help='whether to randomly initiate drone start', metavar='')
    parser.add_argument('--guard_duration',     default=DEFAULT_GUARD_DURATION,   type=int,        help='How many timesteps to keep guard active after switch', metavar='')
    parser.add_argument('--safety_threshold',   default=DEFAULT_SAFETY_THRESHOLD, type=float,      help='threshold q value for triggering guard activation' ,metavar='')
    parser.add_argument('--safe_target',        default=DEFAULT_SAFE_TARGET,      type=str2bool,   help='whether to safely initiate drone target', metavar='')
    parser.add_argument('--bias_target',        default=DEFAULT_BIAS_TARGET,      type=str2bool,   help='whether to bias initiate drone target', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

            


        
        
        



