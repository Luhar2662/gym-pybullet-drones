import numpy as np
import torch as th

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ISAACSFilteredHoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position, with an SAC or ISAACS model as a filter"""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 fallback_model = None,
                 fallback_threshold: float = 0.25,
                 terminate_on_boundary: bool = False,
                 ):
        """

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 8
        self.TERMINATE_ON_BOUNDARY = terminate_on_boundary
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
        self.model = fallback_model
        self.fallback_threshold = fallback_threshold
        self.filter_activations = 0

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        return ret

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
        if self.TERMINATE_ON_BOUNDARY:
            # 2x2x2 box centered on TARGET_POS; floor threshold >0 since physics prevents z<0
            if (abs(state[0] - self.TARGET_POS[0]) > 1.0 or
                abs(state[1] - self.TARGET_POS[1]) > 1.0 or
                state[2] > self.TARGET_POS[2] + 1.0 or
                state[2] < 0.05):
                return True
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
        if not self.TERMINATE_ON_BOUNDARY:
            if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
                 # or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
            ):
                return True
        # else:
        #     if abs(state[7]) > .4 or abs(state[8]) > .4: # Truncate when the drone is too tilted
        #         return True
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
    
    ###############################################################################

    def reset(self, **kwargs):
        self.filter_activations = 0
        return super().reset(**kwargs)

    ###############################################################################

    def step(self, candidate):
        '''
        Override system step function to filter out unsafe actions according to the model.
        '''
        if self.model is None:
            return super().step(candidate)

        passthrough = candidate
        obs = self._computeObs()

        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        act_t = th.as_tensor(candidate, device=self.model.device).float().reshape(1, -1)

        with th.no_grad():
            dist_t = self.model.disturbance_actor._predict(obs_t, deterministic=True)
            q1, q2 = self.model.policy.critic(obs_t, act_t, dist_t)
        q_min = min(q1.item(), q2.item())

        if q_min < self.fallback_threshold:
            self.filter_activations += 1
            safe_action, _ = self.model.predict(obs, deterministic=True)
            passthrough = safe_action.reshape(1, 4).astype(np.float64)

        return super().step(passthrough)

