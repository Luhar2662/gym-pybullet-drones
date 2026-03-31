import numpy as np

import torch as th

import pybullet as p

import gymnasium as gym

import pybullet_data
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import time

class DisturbedEnvWrapper(gym.Wrapper):
    def __init__(self, env, disturbance_bound = .2):
        super().__init__(env)
        self.disturbance_bound = disturbance_bound
    
    def step(self, action):
        noise = np.random.uniform(-self.disturbance_bound, self.disturbance_bound, size=action.shape)
        disturbed = np.clip(action+noise, self.action_space.low, self.action_space.high)
        return self.env.step(disturbed)

class SafeHoverAviary(BaseRLAviary):
    """Variation of the single-agent hover environment, using safety margins for rewards for compatibility with SafetySAC."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([0,0,.1]).reshape(1,3),
                 initial_rpys=np.zeros((1,3)),
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 num_segments = 1,
                 segment_path = False,
                 warmup_dur = 3,
                 random_init = False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 biased_random = True,
                 bias_threshold = .5,
                 random_eval = False,
                 fallback_model = None,
                 random_vel = False,
                 vel_range = 1.0,
                 ang_vel_range = 1.0,
                 hover_threshold = 0.8,
                 hover_steps = 30,
                 episode_len_sec = 6,
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        biased_random: bool, optional
            Decides whether to use biased_random_target() when initializing from random_init
        bias_threshold : float, optional
            The percentage chance to sample uniformly as opposed to from the biased intervals when using biased_random_target()
        fallback_model : SafetySAC Model, optional
            For use with safe initialization and filtered training
        random_eval: bool, optional
            Decides whether to use random_eval_target() when initializing episodes
        fallback_model: ISAACS model, optional
        
        random_vel: bool, optional

        vel_range: float, optional

        ang_vel_range: float, optional

        hover_threshold: float, optional

        hover_steps: int, optional

        episode_len_sec: int, optional

        """
        self.NUM_SEGMENTS = num_segments
        self.warmup_called = 0
        self.WARMUP_DUR = warmup_dur
        self.RANDOM_INIT = random_init
        self.RANDOM_VEL = random_vel
        self.VEL_RANGE = vel_range
        self.ANG_VEL_RANGE = ang_vel_range
        self.HOVER_THRESHOLD = hover_threshold
        self.HOVER_STEPS = hover_steps
        self.hover_counter = 0
        self.SEGMENT_PATH = segment_path
        self.RANDOM_TARGET = None
        self.RANDOM_TARGET_RPY = None
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = episode_len_sec
        self.BIASED_RANDOM = biased_random,
        self.BIAS_THRESHOLD = bias_threshold
        self.RANDOM_EVAL = random_eval
        self.FALLBACK_MODEL = fallback_model
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        
        #return -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2
        #ground_margin = np.linalg.norm(state[0:3]-np.array([0,0,0]))
        #ceil_margin = np.linalg.norm(np.array([0,0,2])-state[0:3])
        # basic way to get signed distance: difference in z-coords. should work even in the not 1-d case, but need to add reward shaping to avoid lateral drift
        ground_margin = state[2]
        left_margin = 1 - state[1]
        right_margin = state[1] + 1
        front_margin = 1 - state[0]
        back_margin = state[0] + 1
        ceil_margin = 2 - state[2]
        safety_margin = min(ground_margin, ceil_margin, back_margin, front_margin, left_margin, right_margin)

        #test reward to penalize being on ground:
        penalty = -20 if state[2]<.25 else 0


        return safety_margin #+ penalty

        #ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        #return ret

    ################################################################################
    def _computeObsForControl(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out OR if the drone has been hovering for a second

        """
        #Hover check -- truncates if the drone has been hovering to make training more efficient
        state = self._getDroneStateVector(0)
        ground_margin = state[2]
        left_margin = 1 - state[1]
        right_margin = state[1] + 1
        front_margin = 1 - state[0]
        back_margin = state[0] + 1
        ceil_margin = 2 - state[2]
        margin = min(ground_margin, ceil_margin, back_margin, front_margin, left_margin, right_margin)

        if margin >= self.HOVER_THRESHOLD:
            self.hover_counter += 1
        else:
            self.hover_counter = 0

        #disable hover truncation for now
        #if self.hover_counter >= self.HOVER_STEPS:
            #return True
            
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years


    ################################################################################

    def segment(self, init_xyzs, goal_pos, segments) -> np.ndarray:
        '''Returns interim waypoints for pid manual warmup during _warmup()
        '''
        target_waypoints = np.zeros((segments, 3))
        tot_offset = goal_pos - init_xyzs

        #split the offset into <segments> amount of offsets
        seg_offset = tot_offset / segments

        #generate the path 
        target_waypoints[0] = init_xyzs
        for i in range(segments):
            target_waypoints[i] = target_waypoints[i-1] + seg_offset
    
        return np.array(target_waypoints)

    ################################################################################

    def _warmup(self, target_pose = None):
        """Runs at the start of each episode (call during reset()). Moves agent to 
        a randomized setpoint to ensure wide distribution of initializations for training
        Parameters
        ----------
        target_pose : np.ndarray, optional (if a specific init setpoint is desired, rather
        than randomized target poses. Default = None.)
        """

        #TODO add support for specific target_pose
        if target_pose is not None:
            target = target_pose
        else:
            #random init within: x in (-1,1), y in (-1,1), z in (.25,2)

            #check if precomputed: if so, simply load target. If not (first run), manually move to target

            if self.RANDOM_TARGET is None:
                control_loop = True
                target = np.array([[np.random.uniform(-1,1),
                                np.random.uniform(-1,1),
                                np.random.uniform(.1, 2)] for i in range(self.NUM_DRONES)])
                target_rpy = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(self.NUM_DRONES)])
            else:
                control_loop = False
                target = self.RANDOM_TARGET
                target_rpy = self.RANDOM_TARGET_RPY
            
        #print("starting warmup!! target: ", target, " ", target_rpy, " warmup number: ", self.warmup_called)
        self.warmup_called += 1
        if control_loop:
            print("calling control loop (first run only)")
            
            waypoints = np.array([self.segment(self.INIT_XYZS[i], target[i], self.NUM_SEGMENTS) for i in range(self.NUM_DRONES)])
            #print("waypoints: ", waypoints.shape)
            waypoint_ct = 0
            waypoint_dur = self.WARMUP_DUR / self.NUM_SEGMENTS
            
            
            #TODO: Use ctrl_aviary or _nextstep() calcs to get to setpoint.
            ctrl_client = DSLPIDControl(drone_model = self.DRONE_MODEL)
            ctrl_client = [DSLPIDControl(drone_model = self.DRONE_MODEL) for i in range(self.NUM_DRONES)]


            #print(warmup called) -- verified that this runs each ep.
            #useful env params: initial_xyzs, initial_rpys, pyb_freq, self.PYB_STEPS_PET_CTRL, self.CLIENT (phys client)

            action = np.zeros((self.NUM_DRONES, 4))
        
        
            first_step = True
            for i in range(0, int(self.WARMUP_DUR*self.CTRL_FREQ)):
                #step the env:
                #print("calling step with action: ", action)
            
            
                if i != 0 and (i % (waypoint_dur*self.CTRL_FREQ)==0) and self.SEGMENT_PATH:
                    waypoint_ct = waypoint_ct + 1
        

                obs, reward, terminated, truncated, info = self.step(action)
                #print("Observation given back: ", obs)
                control_obs = self._computeObsForControl()
                #print("env observation: ", obs)
                #print("control obs: ", control_obs)
                #time.sleep(30)

                
                if first_step:
                    print("starting position: ", control_obs[0])
                    first_step = False
            
            


                #### Compute control for the current way point #############
        
                for j in range(self.NUM_DRONES):
                    if self.SEGMENT_PATH:
                        target_wp = waypoints[j, waypoint_ct, :]
                    else:
                        target_wp = np.hstack([target[j,0:2], target[j, 2]])
                    #print("target shape: ", np.hstack([target[j,0:2], target[j, 2]]))

                    #for i in range(num_drones):
                        #print("waypoints: ", segment(INIT_XYZS[i], target[0], num_segments))
                        #current waypoint: waypoints[waypoint_ct, j, 0:2]
            

                    action[j, :], _, _ = ctrl_client[j].computeControlFromState(control_timestep=self.CTRL_TIMESTEP,
                                                                    state=control_obs[j],
                                                                    target_pos=target_wp,
                                                                    #target_pos=np.hstack([target[j,0:2], target[j, 2]]),
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=self.INIT_RPYS[j, :]
                                                                    )
                #skip logging
        
        
        #pre-compute next target!

        if self.RANDOM_EVAL:
            self.RANDOM_TARGET, self.RANDOM_TARGET_RPY = self.eval_random_target()
        elif self.BIASED_RANDOM:
            self.RANDOM_TARGET, self.RANDOM_TARGET_RPY = self.biased_random_target()
        else:
            self.RANDOM_TARGET, self.RANDOM_TARGET_RPY = self.random_target()
        
        

        self.INIT_XYZS = self.RANDOM_TARGET
        self.INIT_RPYS = self.RANDOM_TARGET_RPY
    
    ################################################################################

    def random_target(self):

        random_target = np.array([[np.random.uniform(-1,1),
                               np.random.uniform(-1,1),
                               np.random.uniform(.05, 2)] for i in range(self.NUM_DRONES)])
        random_target_rpy = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(self.NUM_DRONES)])
        
        return random_target, random_target_rpy
    
    ################################################################################

    def biased_random_target(self):

        intervals = [[-1.0,-.5],[.5,1.0]]
        z_intervals = [[.05,.5],[1.5,2.0]]

        bias = np.random.rand()
        if bias > self.BIAS_THRESHOLD:
            x_int = np.random.choice([0,1])
            y_int = np.random.choice([0,1])
            z_int = np.random.choice([0,1])

            random_target = np.array([[np.random.uniform(intervals[x_int][0],intervals[x_int][1]),
                               np.random.uniform(intervals[y_int][0],intervals[y_int][1]),
                               np.random.uniform(z_intervals[z_int][0],z_intervals[z_int][1])] for i in range(self.NUM_DRONES)])
        else:
            random_target = np.array([[np.random.uniform(-1,1),
                               np.random.uniform(-1,1),
                               np.random.uniform(.05, 2)] for i in range(self.NUM_DRONES)])

        random_target_rpy = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(self.NUM_DRONES)])
        
        return random_target, random_target_rpy
    
    ################################################################################

    def eval_random_target(self):

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
                               np.random.uniform(z_intervals[z_int][0],z_intervals[z_int][1])] for i in range(self.NUM_DRONES)])

        random_target_rpy = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(self.NUM_DRONES)])
        
        return random_target, random_target_rpy
    
    ################################################################################

    def safe_random_init(self):
        if self.FALLBACK_MODEL is not None:
            verified = False
    
            while not verified:
                candidate_pos, candidate_rpy = self.eval_random_target()

                #check viability using a helper env
                test_env = SafeHoverAviary(gui=False, obs=self.OBS_TYPE, act=self.ACT_TYPE, initial_xyzs=candidate_pos, initial_rpys= candidate_rpy)

                obs, _ = test_env.reset(seed=42, options={})
                action, _ = self.FALLBACK_MODEL.predict(obs, deterministic=True)

                obs_t, _ = self.FALLBACK_MODEL.policy.obs_to_tensor(obs)
                act_t = th.as_tensor(action, device=self.FALLBACK_MODEL.device).float()

                with th.no_grad():
                    q1, q2 = self.FALLBACK_MODEL.policy.critic(obs_t, act_t)

                q_avg = np.mean([q1.item(), q2.item()])

                if q_avg > 0:
                    verified = True
                else: 
                    print("unsafe config: ", candidate_pos, ", resampling!")
    
            #verified safe, so pass on!

            return candidate_pos, candidate_rpy
        
        else:
            raise ValueError("Cannot call safe init without passing a fallback model!")

    ################################################################################

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment. Overrides reset in BaseAviary.py

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Warmup for RL methods -- move to a random setpoint. Modification from BaseAviary
        #print('random init: ', self.RANDOM_INIT)
        if self.RANDOM_INIT:
            self._warmup()
        self.hover_counter = 0  # reset AFTER warmup so PID traversal doesn't pre-fill the counter
        if self.RANDOM_VEL:
            lin_vel = np.random.uniform(-self.VEL_RANGE, self.VEL_RANGE, size=3)
            ang_vel = np.random.uniform(-self.ANG_VEL_RANGE, self.ANG_VEL_RANGE, size=3)
            p.resetBaseVelocity(self.DRONE_IDS[0],
                                linearVelocity=lin_vel,
                                angularVelocity=ang_vel,
                                physicsClientId=self.CLIENT)
            self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    def get_warmup_number(self):
        return self.warmup_called