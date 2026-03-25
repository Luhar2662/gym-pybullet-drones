"""Alternative RL algorithm implementing ISAACS variation of SAC, based on the 
SB3 implementation of SAC (Soft Actor-Critic)

Off-policy algorithm training a control policy and an adversarial "disturbance" policy

Critic is updated using the safety Bellman Backip:
"""

from typing import Any, ClassVar, Optional, TypeVar, Union, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, TrainFreq, TrainFreqUnit, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, should_collect_more_steps
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv
from gym_pybullet_drones.isaacs.isaacs_utils import (
    ContinuousISAACSCritic,
    DisturbanceActor,
    ISAACSLeaderboard,
    ISAACSPolicy,
    ISAACSReplayBuffer,
)


class ISAACS(SAC):
    """
    Iterative Soft Adverserial Actor Critic for Safety (ISAACS)
    Off-Policy Variation to Soft Actor Critic,
    This implementation extends upon the Stable Baselines 3.0 implementation of SAC,
    with the proposed ISAACS framework defined in:
    Paper: https://arxiv.org/abs/2212.03228
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    New Params:
    :param disturbance_space: The disturbance space for the DisturbanceActor
    :param disturbance_ent_coef:
    :param actor_update_interval:
    """
    
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # assign policy-name strings such that we create the right critics and actors with _build()
    policy: ISAACSPolicy
    actor: Actor
    disturbance_actor: DisturbanceActor
    critic: ContinuousISAACSCritic
    critic_target: ContinuousISAACSCritic


    #default parameters taken from SB3's SAC implementation
    def __init__(
            self,
            policy: Union[str, type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, tuple[int,str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            n_steps: int = 1,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            disturbance_space: spaces.Box = None,
            disturbance_ent_coef: float = 1.0,
            actor_update_interval: int = 1,
    ):
        #load IsaacsReplayBuffer unless a custom buffer has been passed
        if replay_buffer_class is None:
            replay_buffer_class = ISAACSReplayBuffer
        
        if policy_kwargs is None:
            policy_kwargs = {}
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.disturbance_space = disturbance_space #even if None, _setup_model will handle this
        self.disturbance_ent_coef = disturbance_ent_coef
        self.actor_update_interval = actor_update_interval

        policy_kwargs["disturbance_space"] = disturbance_space
        replay_buffer_kwargs["disturbance_space"] = disturbance_space

        super().__init__(
                policy,
                env,
                learning_rate,
                buffer_size,
                learning_starts,
                batch_size,
                tau,
                gamma,
                train_freq,
                gradient_steps,
                action_noise,
                replay_buffer_class,
                replay_buffer_kwargs,
                optimize_memory_usage,
                n_steps,
                ent_coef,
                target_update_interval,
                target_entropy,
                use_sde,
                sde_sample_freq,
                use_sde_at_warmup,
                stats_window_size,
                tensorboard_log,
                policy_kwargs,
                verbose,
                seed,
                device,
                _init_setup_model,
            )

        #Create the additional adversary policy (pi_d)
    
    def _setup_model(self) -> None:
        super()._setup_model()   
    # _create_aliases will be called by SAC's version of _setup_model(); overriding _create_aliases below

    def _create_aliases(self) -> None:
        '''
        Overriding _create_aliases(). Additionally need to create the disturbance target alias
        and initialize the leaderboard.
        '''
        super()._create_aliases()
        self.disturbance_actor = self.policy.disturbance_actor
        self.leaderboard = ISAACSLeaderboard(self.disturbance_actor)
    

    # Data collection and replay buffer sampling methods:

    def _sample_disturbance(
            self,
            learning_starts: int,
            n_envs: int,
            adversary: Optional[DisturbanceActor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Sample disturbance from the given adversary (defaults to self.disturbance_actor).
        Returns:
        - disturbance (shape: (n_envs, disturbance_dim) in env scale)
        - buffer_disturbance (shape: (n_envs, disturbance_dim) scaled to [-1,1])

        The adversary argument allows collect_rollouts to pass the per-episode disturbance
        policy sampled from the leaderboard rather than always using self.disturbance_actor.
        '''
        if adversary is None:
            adversary = self.disturbance_actor
        d_space = self.policy.disturbance_space

        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled = np.array([d_space.sample() for _ in range(n_envs)])
        else:
            obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
            with th.no_grad():
                unscaled = (
                    adversary._predict(obs_tensor, deterministic=False)
                    .cpu()
                    .numpy()
                )
        if isinstance(d_space, spaces.Box):
            scaled = adversary.scale_action(unscaled)
            scaled = np.clip(scaled, -1, 1)
            return adversary.unscale_action(scaled), scaled
        else:
            return unscaled, unscaled
        
    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        '''
        Override collect_rollouts to sample a disturbance d, and then apply the disturbance additively by default.
        Follows SB3 Method, except with additive disturbance sampling and application.
        '''
        
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "should at least collect one step or episode"

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must only use one env when doing episodic training"

        if self.use_sde:
            self.actor.reset_noise(env.num_envs) #type: ignore[operator]
            self.disturbance_actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        # Sample one episode adversary per env from the leaderboard at the start of each
        # rollout. Re-sampled whenever an episode ends

        # STUB: leaderboard always returns self.disturbance_actor, so this is a no-op for now.
        episode_adversaries = [self.leaderboard.sample_disturbance_policy() for _ in range(env.num_envs)]

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)  # type: ignore[operator]
                self.disturbance_actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # ISAACS: sample adversarial disturbance using the per-episode leaderboard disturbance.
            # NOTE: stub leaderboard always returns self.disturbance_actor, so all identical for now
            # For simplicity with multiple envs, use env 0's adversary for the full batch.

            # TODO: when leaderboard is real, handle per-env adversaries properly.
            disturbances, buffer_disturbances = self._sample_disturbance(
                learning_starts, env.num_envs, adversary=episode_adversaries[0]
            )

            if isinstance(self.action_space, spaces.Box) and isinstance(self.policy.disturbance_space, spaces.Box):
                disturbed_actions = np.clip(
                    actions + disturbances,
                    self.action_space.low,
                    self.action_space.high
                )
            else: #For now, if disturbances are not an additive continuous space, do nothing. Overwrite as needed 
                disturbed_actions = actions

            if self._vec_normalize_env is not None:                                                                             
                self._last_original_obs = self._vec_normalize_env.get_original_obs()


            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(disturbed_actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation, + disturbance)
            self._last_buffer_disturbances = buffer_disturbances
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)    

            self._last_obs = new_obs
            self._last_episode_starts = dones

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Sample a new episode adversary for this env from the leaderboard
                    episode_adversaries[idx] = self.leaderboard.sample_disturbance_policy()

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)


    def _store_transition(self,
            replay_buffer, 
            buffer_action, 
            new_obs, 
            reward, 
            dones, 
            infos
    ):     
        '''
        Override _store_transitions to support adding disturbances in ISAACSReplayBuffer instances
        '''
        from copy import deepcopy
                                                                                                                        
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()                                                       
            reward_ = self._vec_normalize_env.get_original_reward()
        else:                                                                                                           
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
                                                                                                                        
        # Handle terminal observations (VecEnv auto-resets, need true final obs)                                        
        next_obs = deepcopy(new_obs_)
        for i, done in enumerate(dones):                                                                                
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):                                                                          
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:                                                             
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():                                                                         
                        next_obs[key][i] = next_obs_[key]
                else:                                                                                                   
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i])                              
    
        replay_buffer.add(                                                                                              
            self._last_original_obs,
            next_obs,
            buffer_action,
            self._last_buffer_disturbances,                                                                             
            reward_,
            dones,                                                                                                      
            infos,  
        )

    
    ########## TRAINING #####################################

    def train(self, gradient_steps: int):
        pass