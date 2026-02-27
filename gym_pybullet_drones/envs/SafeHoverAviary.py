import numpy as np

import pybullet as p
import pybullet_data
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class SafeHoverAviary(BaseRLAviary):
    """Variation of single agent hover env -- uses margin for reward"""

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
                 act: ActionType=ActionType.RPM
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

        """
        self.NUM_SEGMENTS = num_segments
        self.warmup_called = 0
        self.WARMUP_DUR = warmup_dur
        self.RANDOM_INIT = random_init
        self.SEGMENT_PATH = segment_path
        self.RANDOM_TARGET = None
        self.RANDOM_TARGET_RPY = None
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
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
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        #if (abs(state[0]) > 1 or abs(state[1]) > 1 or state[2] > 2.0 # Truncate when the drone is too far away
        #     or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        #):
        #    return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
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
            
        print("starting warmup!! target: ", target, " ", target_rpy, " warmup number: ", self.warmup_called)
        self.warmup_called += 1
        if control_loop:
            print("calling control loop (first run only)")
            
            waypoints = np.array([self.segment(self.INIT_XYZS[i], target[i], self.NUM_SEGMENTS) for i in range(self.NUM_DRONES)])
            #print("waypoints: ", waypoints.shape)
            waypoint_ct = 0
            waypoint_dur = self.WARMUP_DUR / self.NUM_SEGMENTS
            
            #print("target: ", target)
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

                #PROBLEM -- DESYNCED / LAGGING BY ONE. CUSTOM ASSIGNMENT IS WORKING THOUGH!
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
        self.RANDOM_TARGET = np.array([[np.random.uniform(-1,1),
                               np.random.uniform(-1,1),
                               np.random.uniform(.05, 2)] for i in range(self.NUM_DRONES)])
        self.RANDOM_TARGET_RPY = np.array([[np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            np.random.uniform(-np.pi/3,np.pi/3),
                            ] for i in range(self.NUM_DRONES)])
        self.INIT_XYZS = self.RANDOM_TARGET
        self.INIT_RPYS = self.RANDOM_TARGET_RPY



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
        print('random init: ', self.RANDOM_INIT)
        if self.RANDOM_INIT:
            self._warmup()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    def get_warmup_number(self):
        return self.warmup_called