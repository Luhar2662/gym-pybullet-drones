import numpy as np
import torch as th
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ISAACSFilteredHoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position, with an ISAACS_tauGDA model as a filter.

    Supports two filter modes controlled by the `horizon` parameter:
      horizon == 0 : standard one-step Q-value check (original behaviour).
      horizon  > 0 : predictive rollout — simulates `horizon` steps forward using
                     PyBullet save/restore, checking bounds and/or Q-values at each
                     future state before committing to the candidate action.
    """

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
                 fallback_model=None,
                 fallback_threshold: float = 0.25,
                 terminate_on_boundary: bool = False,
                 horizon: int = 0,
                 check_q: bool = True,
                 check_bounds: bool = True,
                 ):
        self.TARGET_POS = np.array([0, 0, 1])
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
        self.horizon = horizon
        self.check_q = check_q
        self.check_bounds = check_bounds
        self.filter_activations = 0
        self.safety_violations = 0
        self.episode_count = 0

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
                self.safety_violations += 1
                print("safety violation occurred")
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
        self.episode_count += 1
        return super().reset(**kwargs)

    ###############################################################################

    def _predictive_safety_check(self, candidate):
        """Roll the simulation forward `self.horizon` steps to check for future safety
        violations without committing to the real environment.

        Step 0 uses `candidate`; subsequent steps use the ISAACS fallback policy.
        PyBullet save/restore ensures no real state is modified.

        Parameters
        ----------
        candidate : np.ndarray
            The candidate action proposed by the learning policy.

        Returns
        -------
        safe : bool
            True if no violation was predicted within the horizon.
        failure_type : str
            Description of the first predicted violation (empty string if safe).
        """
        p_client = self.getPyBulletClient()
        saved_state = p.saveState(physicsClientId=p_client)
        saved_step_counter = self.step_counter
        saved_last_clipped = self.last_clipped_action.copy() if hasattr(self, 'last_clipped_action') else None
        saved_safety_violations = self.safety_violations

        violated = False
        failure_type = ""
        current_action = candidate.copy()

        try:
            for step_k in range(self.horizon):
                # Step physics only (parent class, bypassing this filter)
                super(ISAACSFilteredHoverAviary, self).step(current_action)

                sim_state = self._getDroneStateVector(0)
                sim_obs = self._computeObs()

                # Bounds check
                if self.check_bounds:
                    if (abs(sim_state[0] - self.TARGET_POS[0]) > 1.0 or
                        abs(sim_state[1] - self.TARGET_POS[1]) > 1.0 or
                        sim_state[2] > self.TARGET_POS[2] + 1.0 or
                        sim_state[2] < 0.05):
                        violated = True
                        failure_type = f"bounds_violation at lookahead step +{step_k + 1}"
                        break

                # Q-value check at the future state using the ISAACS fallback action
                if self.check_q:
                    obs_t, _ = self.model.policy.obs_to_tensor(sim_obs)
                    fallback_action, _ = self.model.predict(sim_obs, deterministic=True)
                    act_t = th.as_tensor(fallback_action, device=self.model.device).float().reshape(1, -1)
                    with th.no_grad():
                        dist_t = self.model.disturbance_actor._predict(obs_t, deterministic=True)
                        q1, q2 = self.model.policy.critic(obs_t, act_t, dist_t)
                    q_mean = np.mean([q1.item(), q2.item()])
                    if q_mean < self.fallback_threshold:
                        violated = True
                        failure_type = f"q_violation at lookahead step +{step_k + 1} (q={q_mean:.4f})"
                        break

                # Use ISAACS fallback for the next simulated step
                if step_k < self.horizon - 1:
                    current_action, _ = self.model.predict(sim_obs, deterministic=True)

        finally:
            p.restoreState(stateId=saved_state, physicsClientId=p_client)
            p.removeState(saved_state, physicsClientId=p_client)
            self.step_counter = saved_step_counter
            if saved_last_clipped is not None:
                self.last_clipped_action = saved_last_clipped
            self.safety_violations = saved_safety_violations
            self._updateAndStoreKinematicInformation()

        return not violated, failure_type

    ###############################################################################

    def step(self, candidate):
        """Override step to filter unsafe actions using the ISAACS fallback model.

        When horizon == 0: standard one-step Q-value check on the candidate action.
        When horizon  > 0: predictive rollout check; falls back to one-step Q-check
                           only if horizon is explicitly 0.
        """
        if self.model is None:
            return super().step(candidate)

        obs = self._computeObs()

        if self.horizon > 0:
            safe, reason = self._predictive_safety_check(candidate)
            if not safe:
                self.filter_activations += 1
                safe_action, _ = self.model.predict(obs, deterministic=True)
                return super().step(safe_action.reshape(1, 4).astype(np.float64))
            return super().step(candidate)

        # horizon == 0: original one-step Q-value check
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        act_t = th.as_tensor(candidate, device=self.model.device).float().reshape(1, -1)
        with th.no_grad():
            dist_t = self.model.disturbance_actor._predict(obs_t, deterministic=True)
            q1, q2 = self.model.policy.critic(obs_t, act_t, dist_t)
        q_min = min(q1.item(), q2.item())

        if q_min < self.fallback_threshold:
            self.filter_activations += 1
            safe_action, _ = self.model.predict(obs, deterministic=True)
            return super().step(safe_action.reshape(1, 4).astype(np.float64))

        return super().step(candidate)

    def print_violations(self):
        print(f"episodes: {self.episode_count - 1} | safety violations: {self.safety_violations}")
