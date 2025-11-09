"""Alternative RL algorithm implementing ISAACS variation of SAC, based on the 
SB3 implementation of SAC (Soft Actor-Critic)"""

from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3 import SAC

class ISAACS(SAC):
    """
    Iterative Soft Adverserial Actor Critic for Safety (ISAACS)
    Off-Policy Variation to Soft Actor Critic,
    This implementation extends upon the Stable Baselines 3.0 implementation of SAC,
    with the proposed ISAACS framework defined in:
    Paper: https://arxiv.org/abs/2212.03228
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
    
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: ISAACSPolicy
    actor: Actor
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
    ):
        super(ISAACS, self).__init__(
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

        


        #MAKE CHANGES HERE if using different critic networks with disturbance actors

    def _create_aliases(self) -> None:
        #rewrite this -- SAC creates aliases for actor, critic, critic targer -- need to add pi_d
        pass

    def train(self, graident_steps: int, batch_size: int = 64) -> None:
        pass
        #need to add training details for adversary

        #TODO Add disturbance action 
        #TODO modify critic input (Q(x,u,d))
        #TODO Modify Q computation
        #TODO Modify actor loss
        #TODO train adversary to minimize Q