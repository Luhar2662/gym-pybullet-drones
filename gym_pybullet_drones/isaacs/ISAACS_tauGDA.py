"""ISAACS with tau-GDA actor update rule from MAGICS
"""

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import SACPolicy

from gym_pybullet_drones.isaacs.ISAACS import ISAACS


class ISAACS_tauGDA(ISAACS):
    """
    ISAACS with the tau-GDA actor update rule from MAGICS (Wang et al., WAFR 2024).

    New parameter vs. base ISAACS:
    :param tau_a: Disturbance timescale ratio. 
    """

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
            train_freq: Union[int, tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[type[ReplayBuffer]] = None,
            replay_buffer_kwargs: Optional[dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
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
            tau_a: float = 1.0,
            leaderboard_eval_env=None,
            leaderboard_update_freq: int = 0,
            leaderboard_k_u: int = 5,
            leaderboard_k_d: int = 5,
            leaderboard_n_eps: int = 5,
    ):
        self.tau_a = tau_a
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
            disturbance_space,
            disturbance_ent_coef,
            actor_update_interval,
            leaderboard_eval_env=leaderboard_eval_env,
            leaderboard_update_freq=leaderboard_update_freq,
            leaderboard_k_u=leaderboard_k_u,
            leaderboard_k_d=leaderboard_k_d,
            leaderboard_n_eps=leaderboard_n_eps,
        )

    def train(self, gradient_steps: int, batch_size: int = 256):
        """
        tau-GDA training loop (MAGICS Alg. 1).

        Per gradient step:
          1. Critic update (safety Bellman backup) — unchanged from base ISAACS.
          2. Actor update 
          3. Disturbance update 
        """
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer, self.disturbance_actor.optimizer]

        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, disturbance_losses, critic_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            #update ent-coefficient following SB3 methods

            if self.ent_coef_optimizer is not None:
                with th.no_grad():
                    _, log_prob = self.actor.action_log_prob(replay_data.observations)
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            ############## Compute Bellman Backup ##################

            #Following SafetySAC implementation from Safe Robotics Labs' SafetyStableBaselines Repo!
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_disturbances, _ = self.disturbance_actor.action_log_prob(replay_data.next_observations)
                
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions, next_disturbances),
                    dim=1,
                )

                # Conservative estimate across Q-networks
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # Entropy term (soft safety value, following SafetySAC / ISAACS paper)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

                # g' = safety margin at next state (stored as reward in buffer)
                g_prime = replay_data.rewards
                not_done = 1.0 - replay_data.dones
                v_to_go = th.minimum(g_prime, next_q_values)

                target_q_values = (1.0 - self.gamma * not_done) * g_prime + self.gamma * not_done * v_to_go

            #################### Critic Update ######################
            current_q_values = self.critic(replay_data.observations, replay_data.actions, replay_data.disturbances)
            
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            #update critic optimizer
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

             # Polyak update of target critic (follows SafetySAC)
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            ############## Actor update (tau-GDA) ##############################
            # Minimizing this maximizes Q (safety) + entropy bonus
            
            if gradient_step % self.actor_update_interval == 0:
                total_step = self._n_updates + gradient_step
                if (self.leaderboard_update_freq > 0
                        and total_step > 0
                        and total_step % self.leaderboard_update_freq == 0):
                    self.leaderboard.update(self.actor, self.disturbance_actor)

                #sample again (this happens earlier in safety_sac, placed here for readability)
                u_pi, u_log_prob = self.actor.action_log_prob(replay_data.observations)
                with th.no_grad():
                    d_pi_for_actor, _ = self.disturbance_actor.action_log_prob(replay_data.observations)

                q_values_u = th.cat(
                    self.critic(replay_data.observations, u_pi, d_pi_for_actor),
                    dim=1,
                )
                min_q_u, _ = th.min(q_values_u, dim=1, keepdim=True)

                actor_loss = (ent_coef * u_log_prob - min_q_u).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            ############## Disturbance Update (tau-GDA) ########################
            # Minimizing this drives Q down (adversarial) + entropy bonus
            #Sample a fresh disturbance from current adversary policy!
            d_pi, d_log_prob = self.disturbance_actor.action_log_prob(replay_data.observations)
            with th.no_grad():
                u_pi_for_dist, _ = self.actor.action_log_prob(replay_data.observations)

            q_values_d = th.cat(
                self.critic(replay_data.observations, u_pi_for_dist, d_pi),
                dim=1,
            )
            min_q_d, _ = th.min(q_values_d, dim=1, keepdim=True)

            disturbance_loss = (min_q_d + self.disturbance_ent_coef * d_log_prob).mean()
            disturbance_losses.append(disturbance_loss.item())

            self.disturbance_actor.optimizer.zero_grad()
            (self.tau_a * disturbance_loss).backward() #TAU_A scaling 
            self.disturbance_actor.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/disturbance_loss", np.mean(disturbance_losses))
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
