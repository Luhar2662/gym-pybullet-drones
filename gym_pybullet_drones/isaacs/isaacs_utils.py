"""
Redefinition / modifications of SB3 implementations for Actor/Critic networks,
Implementation of ISAACS specific Replay Buffer, and other necessary changes and utility classes
for ISAACS training implementation

Contains:
- DisturbanceActor: same architecture roughly as the general Actor used in SB3 SAC
- ISAACSReplayBuffer and ISAACSReplayBufferSamples: extends replay buffers from SAC to support disturbance inputs
- ContinuousISAACSCritic: extends ContinuousCritic input layer to support disturbance input
- ISAACSPolicy: Extends SB3's SACPolicy to maintain disturbance actor network during training
"""

import copy
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union, Tuple, Type, List

import torch as th
import numpy as np
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.vec_env import VecNormalize
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

class DisturbanceActor(Actor):
    '''
    Adversarial agent policy, using the same architecture as the SAC actor.
    Adversarial training objective is applied externally in the ISAACS train() method.
    
    By default, the disturbance dimension is the same as the action dimension, and the admissable
    disturbance values are smaller (approximating motor noise). This can be overwritten by passing a 
    valid continuous disturbance space (spaces.Box) -- this change must be mirrored in env dynamics.
    '''
    def __init__(
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
            disturbance_space: spaces.Box = None,
    ):
        
        if disturbance_space is not None:
            self.disturbance_space = disturbance_space
        else:
            self.disturbance_space = action_space
        
        self.disturbance_dim = self.disturbance_space.shape[0]

        super().__init__(
            observation_space,
            self.disturbance_space,
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

class ISAACSReplayBufferSamples(NamedTuple):
    '''
    Single transition instance stored in Replay Buffer -- extended from SB3 to include disturbance input
    '''
    observations: th.Tensor
    actions: th.Tensor
    disturbances: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class ISAACSReplayBuffer(ReplayBuffer):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    disturbances: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    '''
    Replay Buffer to be used for ISAACS, modeled closely off of SB3's replay buffer for SAC.
    Main changes include support for storing disturbances and sampling them.
    '''

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            disturbance_space: spaces.Space = None
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
        self.disturbance_dim = self.action_dim if disturbance_space is None else disturbance_space.shape[0]
        if disturbance_space is None:
            self.d_space = action_space
        else:
            self.d_space = disturbance_space

        self.disturbances = np.zeros(
            (self.buffer_size, self.n_envs, self.disturbance_dim), dtype = self._maybe_cast_dtype(self.d_space.dtype)
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

        
        action = action.reshape((self.n_envs, self.action_dim))
        disturbance = disturbance.reshape((self.n_envs, self.disturbance_dim))

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


class ContinuousISAACSCritic(ContinuousCritic):
    '''
    Critic class for ISAACS, which (analogous to SAC) predicts Q-values from (obs, act, disturbance)

    Extends ContinuousCritic by appending disturbance vector to the input layer, and so the Q-network size
    increases by 'disturbance_dim' 
    '''

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            net_arch: List[int],
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            disturbance_space: spaces.Box = None
    ):
        '''
        New input fields:
        disturbance_space: spaces.Box --> the disturbance space used by the disturbance actor in this instance
            of ISAACS. By default is None, in which case the disturbance space is assumed to be identical to the 
            Action Space
        '''

        super().__init__(
            observation_space = observation_space,
            action_space = action_space,
            net_arch = net_arch,
            features_extractor = features_extractor,
            features_dim = features_dim,
            activation_fn = activation_fn,
            normalize_images = normalize_images,
            n_critics = n_critics,
            share_features_extractor = share_features_extractor,
        )

        self.d_space = action_space if disturbance_space is None else disturbance_space
        self.disturbance_dim = get_action_dim(self.d_space)
        self.action_dim = get_action_dim(action_space)

        # rebuild default Q-Networks (from SAC) and extend input size by self.disturbance_dim; all other
        # arguments same as SAC
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            q_net = create_mlp(
                features_dim + self.action_dim + self.disturbance_dim, 1, net_arch, activation_fn
            )
            created_net = nn.Sequential(*q_net)
            self.q_networks.append(created_net)
    
    def forward(
            self,
            obs: th.Tensor,
            actions: th.Tensor,
            disturbances: th.Tensor,
    ) -> Tuple[th.Tensor,...]:
        '''
        adapted from SB3 common policies --> ContinuousCritic, accounting for disturbance input
        '''
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        
        qvalue_input = th.cat([features, actions, disturbances], dim = 1)

        return tuple(qnet(qvalue_input) for qnet in self.q_networks)

    def q1_forward(
            self, 
            obs: th.Tensor,
            actions: th.Tensor,
            disturbances: th.Tensor,
    ) -> th.Tensor:
        '''
        Allows prediction using only the first q_network for easier computation, following implementation
        of ContinuousCritic in SB3.
        '''

        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        
        qvalue_input = th.cat([features, actions, disturbances], dim = 1)

        return self.q_networks[0](qvalue_input)


class ISAACSPolicy(SACPolicy):
    '''
    Extends SB3's SACPolicy for ISAACS training.
    Changes:
    - Adds a DisturbanceActor and accompanying optimizer
    - Overrides make_critic to instantiate a ContinuousISAACSCritic instead
    '''

    actor: Actor #Already set in SACPolicy, just repeated here

    disturbance_actor: DisturbanceActor
    critic: ContinuousISAACSCritic
    critic_target: ContinuousISAACSCritic

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Box,
            lr_schedule: Schedule,
            net_arch: list[int] | dict[str, list[int]] | None = None,
            activation_fn: type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: dict[str, Any] | None = None,
            normalize_images: bool = True,
            optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: dict[str,Any] | None = None,
            n_critics: int = 2,
            share_features_extractor: bool = False,
            disturbance_space: spaces.Box = None,
    ):
        
        '''
        Overloading so that disturbance details can be sorted at the top level 

        New input arguments:
        disturbance_space: spaces.Box --> the disturbance space for the environment, if it exists. By default,
            the disturbance space will be assumed to be identical to the action space
        '''
        if disturbance_space is None:
            self.disturbance_space = action_space
        else:
            self.disturbance_space = disturbance_space

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )
        
        self.disturbance_dim = get_action_dim(self.disturbance_space)
        #or self.disturbance_space.shape[0] --> unsure!

    '''
    Overriding make methods to account for DisturbanceActor and ContinuousISAACSCritic!!
    '''
    def make_disturbance_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        
        #follows SB3's SACPolicy.make_actor(), but overrides kwargs with the disturbance space instead!
        disturbance_kwargs = actor_kwargs.copy()
        disturbance_kwargs["disturbance_space"] = self.disturbance_space

        return DisturbanceActor(**disturbance_kwargs).to(self.device)
    
    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> ContinuousISAACSCritic:         
      critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)                       
      critic_kwargs["disturbance_space"] = self.disturbance_space                                                   
      return ContinuousISAACSCritic(**critic_kwargs).to(self.device)
    

    def _build(self, lr_schedule: Schedule) -> None:
        '''
        Overrides _build() to also instantiate disturbance_actor
        '''

        super()._build(lr_schedule)

        self.disturbance_actor = self.make_disturbance_actor()
        self.disturbance_actor.optimizer = self.optimizer_class(
            self.disturbance_actor.parameters(),
            lr = lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def set_training_mode(self, mode: bool) -> None:
        '''
        Overrides SACPolicy's set_training_mode() to also update disturbance actor
        '''

        super().set_training_mode(mode)
        if hasattr(self, "disturbance_actor"):
            self.disturbance_actor.set_training_mode(mode)


class ISAACSLeaderboard:
    """
    Stub for the ISAACS leaderboard scheme 

    This stub always returns the current live disturbance actor and no-ops on update,
    so the core training loop runs correctly without the leaderboard machinery.
    Replace sample_disturbance_policy() and update() to enable the full scheme.
    """

    def __init__(
            self, 
            disturbance_actor: DisturbanceActor,
            eval_env=None,
            k_u: int = 5,
            k_d: int = 5,
            n_eps: int = 5,
        ):
        '''
        Parameters:
        disturbance_actor: the current disturbance actor being trained, in the event that the leaderboard is not yet filled
        eval_env: the environment used for testing rounds
        k_u: amount of actor networks being stored
        k_d: amount of disturbance networks being stored
        n_eps: number of episodes per round (between each u, d pairing)
        '''
        self.current_disturbance = disturbance_actor
        self.eval_env = eval_env
        self.k_u, self.k_d = k_u, k_d
        self.n_eps = n_eps
        
        #create leaderboards to hold actor and disturbance policies, as well as win rate dict and sampling distributions
        self.actor_policies = []
        self.dist_policies = []

        self.win_rates = {} #key: (d_index, u_index) --> value: safety violation rate (win rate of disturbance / error)
        self.sample_dist = np.array([])


    def sample_disturbance_policy(self) -> DisturbanceActor:
        """
        Sample one disturbance policy for a new episode from P_{Π^d}.
        Called once per episode (or per env reset) in collect_rollouts.

        """
        if len(self.dist_policies) == 0:
            return self.current_disturbance

        idx = int(np.random.choice(len(self.dist_policies), p=self.sample_dist))
        return self.dist_policies[idx]

    def update(self, pi_u: Actor, pi_d: DisturbanceActor) -> None:
        """
        Collect the current actor and disturbance policies, run the tournament, and update leaderboard and samplic dist.
        Do nothing if eval_env is not valid or provided!
        """
        
        if self.eval_env is None:
            return

        u_current = copy.deepcopy(pi_u)
        u_current.set_training_mode(False)
        d_current = copy.deepcopy(pi_d)
        d_current.set_training_mode(False)

        new_d_idx, new_u_idx = len(self.dist_policies), len(self.actor_policies)

        num_actors, num_disturbances = len(self.actor_policies), len(self.dist_policies)

        #test new disturbance against existing controllers
        for actor_idx, actor in enumerate(self.actor_policies):
            win_rate = self.run_evaluation(actor, d_current)
            self.win_rates[(new_d_idx, actor_idx)] = win_rate

        #test new controller against existing disturbances
        for dist_idx, disturbance in enumerate(self.dist_policies):
            win_rate = self.run_evaluation(u_current, disturbance)
            self.win_rates[(dist_idx, new_u_idx)] = win_rate

        #test new pair
        win_rate = self.run_evaluation(u_current, d_current)
        self.win_rates[(new_d_idx, new_u_idx)] = win_rate

        #update actor and disturbance leaderboards
        self.actor_policies.append(u_current)
        self.dist_policies.append(d_current)

        if len(self.actor_policies) > self.k_u: #collate the average win rates against each actor, and remove the one with the highest
            average_win_rates = []
            for u in range(len(self.actor_policies)):
                rates = [self.win_rates.get((d, u), 0.0) for d in range(len(self.dist_policies))]
                average_win_rates.append(np.mean(rates) if rates else 0.0)
            #now, remove the actor with the highest disturbance win rate
            self.remove_actor(int(np.argmax(average_win_rates)))
        
        if len(self.dist_policies) > self.k_d: #same idea with disturbances as above, obviously removing the minimum instead
            m = self.compute_win_rates()
            self.remove_dist(int(np.argmin(m)))
            
        #determine sampling distribution for disturbances
        m = self.compute_win_rates() #calculate m again in case a disturbance was dropped above
        exponential = np.exp(m - np.max(m)) #implement softmax
        self.sample_dist = exponential / exponential.sum()




    def run_evaluation(self, pi_u, pi_d) -> float:
        '''
        Run episodes between the given actor and disturbance networks, and return the fraction of episodes in which
        a safety violation occurs
        '''

        env = self.eval_env
        device = next(pi_u.parameters()).device

        num_violations = 0

        for _ in range(self.n_eps):
            obs, _ = env.reset()
            done = False
            truncated = False
            violation = False

            while not (done or truncated):
                obs_t = th.FloatTensor(obs).unsqueeze(0).to(device)
                with th.no_grad():
                    action = pi_u._predict(obs_t, deterministic=True).cpu().numpy().squeeze(0)
                    disturbance = pi_d._predict(obs_t, deterministic=True).cpu().numpy().squeeze(0)
                # by default, model disturbance as additive
                disturbed = np.clip(action + disturbance, env.action_space.low, env.action_space.high)

                obs, margin, done, truncated, _ = env.step(disturbed)
                
                if margin < 0:
                    violation = True
                    break
            #rollout finished
            if violation:
                num_violations += 1
        return float(num_violations) / float(self.n_eps)
    
    def compute_win_rates(self) -> np.ndarray:
        '''
        Determine the total winrates of each disturbance across all controllers
        '''
        m = []
        for dist in range(len(self.dist_policies)):
            rates = [self.win_rates.get((dist, act), 0.0) for act in range(len(self.actor_policies))]
            m.append(float(np.mean(rates)) if rates else 0.0)
        return np.array(m, dtype=np.float64)

    def remove_actor(self, idx) -> None:
        '''
        Removes the actor at the given index, and reshuffles win_rates accordingly
        '''
        self.actor_policies.pop(idx)
        new_win_rates = {}
        for (d, u), wr in self.win_rates.items():
            if u == idx:
                continue
            new_u = u if u < idx else u - 1 #shift all subsequent actors one step down
            new_win_rates[(d, new_u)] = wr #place new pair in win rates
        
        self.win_rates = new_win_rates
    
    def remove_dist(self, idx) -> None:
        '''
        Removes the disturbance at the given index, and reshuffles win_rates accordingly
        '''
        self.dist_policies.pop(idx)
        new_win_rates = {}
        for (d, u), wr in self.win_rates.items():
            if d == idx:
                continue
            new_d = d if d < idx else d - 1  # shift all subsequent disturbances one step down
            new_win_rates[(new_d, u)] = wr

        self.win_rates = new_win_rates
                




