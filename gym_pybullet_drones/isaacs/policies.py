"""Redefinition / modifications of SB3 implementations for Actor/Critic network,
Implementation of ISAACS specific Replay Buffer, and other necessary changes for ISAACS
training implementation

CHANGES:
- Use the same Actor -- not much to change, just maximizes the critic
- CHANGE Critic and Critic Target-- add disturbance dimension to input
- Create a Disturbance Actor networ


k
- Extend SACPolicy for ISAACSPolicy -> override critic with the safety bellman update and 
additional disturbance knowledge
"""

from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union

import torch as th
import numpy as np
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3 import SAC

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

#TODO extend sac.Actor (?)

#TODO define Disturbance Policy (extends BasePolicy)
class DisturbanceActor(Actor):
    def init(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            net_arch: list[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: type[nn.Module] = nn.ReLU,
            use_sde: bool= False,
            log_std_init: float = -3,
            full_std: bool = True,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = True,
    ):
        super().__init__(
            self,
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )

#TODO Extend ReplayBuffer to include storing disturbance actions
class ISAACSReplayBufferSamples(NamedTuple):
    observations: nn.Tensor
    actions: nn.Tensor
    disturbances: nn.Tensor
    next_observations: nn.Tensor
    dones: nn.Tensor
    rewards: nn.Tensor
    #for n-step replay buffer
    discounts: Optional[nn.Tensor] = None

class ISAACSReplayBuffer(ReplayBuffer):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    disturbances: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs = n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        #add init for disturbance buffer -- for now, model as noise in the action space
        self.disturbances = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype = self._maybe_cast_dtype(action_space.dtype)
        )

    #overload add -> same general logic / code as in RandomBuffer, but with disturbance added
    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            disturbance: np.ndarray,
            reward:np.ndarray,
            done: np.ndarray,
            infos: list[dict[str,Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        disturbance = disturbance.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.disturbances[self.pos] = np.array(disturbance)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def sample(
            self,
            batch_size:int,
            env: Optional[VecNormalize]=None
    ) -> ISAACSReplayBufferSamples:
        """
        Sample tuple from the ISAACSReplayBuffer (RB implementation extended from SB3).
        :param batch_size: Number of transitions to sample
        :param env: gym environment for samples
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples(batch_inds, env=env)
        
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size = batch_size)
        
        return self._get_samples(batch_inds, env=env)
    
    def _get_samples(
            self, 
            batch_inds:np.ndarray,
            env: Optional[VecNormalize]=None
            ) -> ISAACSReplayBufferSamples:
        env_indices = np.random.randint(0, high = self.n_envs, size = (len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env) 
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self.disturbances[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ISAACSReplayBufferSamples(*tuple(map(self.to_torch, data)))

#TODO Extend the full SAC policy to include all 3 networks

#TODO Extend ContinuousCritic -> ContinuousISAACSCritic - maintains two q networks
