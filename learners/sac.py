#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/13
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: sac.py
# =====================================

import logging

import numpy as np

from preprocessor import Preprocessor
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SACLearner(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.batch_size = self.args.replay_batch_size
        self.policy_with_value = policy_cls(**vars(self.args))
        self.batch_data = {}
        self.preprocessor = Preprocessor(self.args.obs_dim, self.args.obs_ptype, self.args.rew_ptype,
                                         self.args.obs_scale, self.args.rew_scale, self.args.rew_shift,
                                         gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.alpha_timer = TimerStat()
        self.target_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        self.counter = 0
        self.num_batch_reuse = self.args.num_batch_reuse

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           }

        with self.target_timer:
            target = self.compute_clipped_double_q_target()

        self.batch_data.update(dict(batch_targets=target,))
        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error(),
                                             rb=rb,
                                             indexes=indexes))

    def compute_clipped_double_q_target(self):
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        act_tp1, logp_tp1 = self.policy_with_value.compute_action(processed_obs_tp1)

        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, act_tp1).numpy()
        target_Q2_of_tp1 = self.policy_with_value.compute_Q2_target(processed_obs_tp1, act_tp1).numpy()

        alpha = self.tf.exp(self.policy_with_value.log_alpha).numpy() if self.args.alpha == 'auto' else self.args.alpha

        clipped_double_q_target = processed_rewards + self.args.gamma * \
                                  (np.minimum(target_Q1_of_tp1, target_Q2_of_tp1)-alpha*logp_tp1.numpy())
        return clipped_double_q_target

    def compute_td_error(self):
        processed_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()  # n_step*obs_dim
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        values_t = self.policy_with_value.compute_Q1(processed_obs, self.batch_data['batch_actions']).numpy()
        target_act_tp1, _ = self.policy_with_value.compute_target_action(processed_obs_tp1)
        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, target_act_tp1).numpy()
        td_error = processed_rewards + self.args.gamma * target_Q1_of_tp1 - values_t
        return td_error

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions, mb_targets):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        with self.tf.GradientTape(persistent=True) as tape:
            with self.tf.name_scope('q_loss') as scope:
                q_pred1 = self.policy_with_value.compute_Q1(processed_mb_obs, mb_actions)
                q_loss1 = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred1 - mb_targets))

                q_pred2 = self.policy_with_value.compute_Q2(processed_mb_obs, mb_actions)
                q_loss2 = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred2 - mb_targets))

        with self.tf.name_scope('q_gradient') as scope:
            q_gradient1 = tape.gradient(q_loss1, self.policy_with_value.Q1.trainable_weights)
            q_gradient2 = tape.gradient(q_loss2, self.policy_with_value.Q2.trainable_weights)

        return q_loss1, q_loss2, q_gradient1, q_gradient2

    @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape() as tape:
            processed_obses = self.preprocessor.tf_process_obses(mb_obs)
            actions, logps = self.policy_with_value.compute_action(processed_obses)
            all_Qs1 = self.policy_with_value.compute_Q1(processed_obses, actions)
            all_Qs2 = self.policy_with_value.compute_Q2(processed_obses, actions)
            all_Qs_min = self.tf.reduce_min((all_Qs1, all_Qs2), 0)
            alpha = self.tf.exp(self.policy_with_value.log_alpha) if self.args.alpha == 'auto' else self.args.alpha
            policy_loss = self.tf.reduce_mean(alpha*logps-all_Qs_min)

            policy_entropy = -self.tf.reduce_mean(logps)
            value_var = self.tf.math.reduce_variance(all_Qs_min)
            value_mean = self.tf.reduce_mean(all_Qs_min)

        with self.tf.name_scope('policy_gradient') as scope:
            policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights,)
            return policy_loss, policy_gradient, policy_entropy, value_mean, value_var

    @tf.function
    def alpha_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape() as tape:
            processed_obses = self.preprocessor.tf_process_obses(mb_obs)
            actions, logps = self.policy_with_value.compute_action(processed_obses)
            log_alpha = self.policy_with_value.log_alpha
            alpha_loss = self.tf.reduce_mean(-log_alpha * self.tf.stop_gradient(logps + self.args.target_entropy))

        with self.tf.name_scope('alpha_gradient') as scope:
            alpha_gradient = tape.gradient(alpha_loss, self.policy_with_value.alpha_model.trainable_weights)
            return alpha_loss, self.tf.exp(log_alpha), alpha_gradient

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        mb_targets = self.batch_data['batch_targets']
        mb_actions = self.batch_data['batch_actions']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.alpha_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="alpha_forward_and_backward", step=0)

    def compute_gradient(self, batch_data, rb, indexes, iteration):  # compute gradient
        if self.counter % self.num_batch_reuse == 0:
            self.get_batch_data(batch_data, rb, indexes)
        self.counter += 1
        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error()))
        mb_obs = self.batch_data['batch_obs']
        mb_actions = self.batch_data['batch_actions']
        mb_targets = self.batch_data['batch_targets']

        with self.q_gradient_timer:
            q_loss1, q_loss2, q_gradient1, q_gradient2 = self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
            q_gradient1, q_gradient_norm1 = self.tf.clip_by_global_norm(q_gradient1, self.args.gradient_clip_norm)
            q_gradient2, q_gradient_norm2 = self.tf.clip_by_global_norm(q_gradient2, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            policy_loss, policy_gradient, policy_entropy, value_mean, value_var = self.policy_forward_and_backward(mb_obs)

        policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                            self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            target_time=self.target_timer.mean,
            q_loss1=q_loss1.numpy(),
            q_loss2=q_loss2.numpy(),
            policy_loss=policy_loss.numpy(),
            policy_entropy=policy_entropy.numpy(),
            mb_targets_mean=np.mean(mb_targets),
            value_mean=value_mean.numpy(),
            value_var=value_var.numpy(),
            q_gradient_norm1=q_gradient_norm1.numpy(),
            q_gradient_norm2=q_gradient_norm2.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
        ))
        if self.args.alpha == 'auto':
            with self.alpha_timer:
                alpha_loss, alpha, alpha_gradient = self.alpha_forward_and_backward(mb_obs)
                alpha_gradient, alpha_gradient_norm = self.tf.clip_by_global_norm(alpha_gradient,
                                                                                  self.args.gradient_clip_norm)
            self.stats.update(dict(alpha=alpha.numpy(),
                                   alpha_loss=alpha_loss.numpy(),
                                   alpha_gradient_norm=alpha_gradient_norm.numpy(),
                                   alpha_time=self.alpha_timer.mean))

            gradient_tensor = q_gradient1 + q_gradient2 + policy_gradient + alpha_gradient
        else:
            gradient_tensor = q_gradient1 + q_gradient2 + policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))

class SACLearnerWithCost(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, args):
        self.args = args
        self.batch_size = self.args.replay_batch_size
        self.policy_with_value = policy_cls(**vars(self.args))
        self.batch_data = {}
        self.preprocessor = Preprocessor(self.args.obs_dim, self.args.obs_ptype, self.args.rew_ptype,
                                         self.args.obs_scale, self.args.rew_scale, self.args.rew_shift,
                                         gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.alpha_timer = TimerStat()
        self.target_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        self.counter = 0
        self.num_batch_reuse = self.args.num_batch_reuse

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           'batch_costs': batch_data[5].astype(np.float32)
                           }

        with self.target_timer:
            target, cost_target = self.compute_clipped_double_q_target()

        self.batch_data.update(dict(batch_targets=target, batch_cost_targets=cost_target))
        if self.args.buffer_type != 'normal':
            td_error, cost_td_error = self.compute_td_error()
            self.info_for_buffer.update(dict(td_error=td_error, cost_td_error=cost_td_error,
                                             rb=rb,
                                             indexes=indexes))

    def compute_clipped_double_q_target(self):
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        act_tp1, logp_tp1 = self.policy_with_value.compute_action(processed_obs_tp1)

        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, act_tp1).numpy()
        target_Q2_of_tp1 = self.policy_with_value.compute_Q2_target(processed_obs_tp1, act_tp1).numpy()

        alpha = self.tf.exp(self.policy_with_value.log_alpha).numpy() if self.args.alpha == 'auto' else self.args.alpha

        clipped_double_q_target = processed_rewards + self.args.gamma * \
                                  (np.minimum(target_Q1_of_tp1, target_Q2_of_tp1)-alpha*logp_tp1.numpy())

        processed_cost = self.batch_data['batch_costs'].numpy() # todo
        target_Q_cost_of_tp1 = processed_cost + self.policy_with_value.compute_Q_cost(processed_obs_tp1, act_tp1).numpy()
        # no gamma for q cost

        return clipped_double_q_target, target_Q_cost_of_tp1

    def compute_td_error(self):
        processed_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()  # n_step*obs_dim
        processed_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()
        processed_cost = self.batch_data['batch_costs'].numpy()# todo
        processed_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        values_t = self.policy_with_value.compute_Q1(processed_obs, self.batch_data['batch_actions']).numpy()
        target_act_tp1, _ = self.policy_with_value.compute_target_action(processed_obs_tp1)
        target_Q1_of_tp1 = self.policy_with_value.compute_Q1_target(processed_obs_tp1, target_act_tp1).numpy()
        td_error = processed_rewards + self.args.gamma * target_Q1_of_tp1 - values_t

        cost_values_t = self.policy_with_value.compute_Q_cost(processed_obs, self.batch_data['batch_actions']).numpy()
        target_Q_cost_of_tp1 = self.policy_with_value.compute_Q_cost(processed_obs_tp1, target_act_tp1).numpy()
        cost_td_error = processed_cost + target_Q_cost_of_tp1 - cost_values_t
        return td_error, cost_td_error

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions, mb_targets, mb_cost_targets):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        with self.tf.GradientTape(persistent=True) as tape:
            with self.tf.name_scope('q_loss') as scope:
                q_pred1 = self.policy_with_value.compute_Q1(processed_mb_obs, mb_actions)
                q_loss1 = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred1 - mb_targets))

                q_pred2 = self.policy_with_value.compute_Q2(processed_mb_obs, mb_actions)
                q_loss2 = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred2 - mb_targets))

                q_pred_cost = self.policy_with_value.compute_Q_cost(processed_mb_obs, mb_actions)
                q_loss_cost = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred_cost - mb_cost_targets))

        with self.tf.name_scope('q_gradient') as scope:
            q_gradient1 = tape.gradient(q_loss1, self.policy_with_value.Q1.trainable_weights)
            q_gradient2 = tape.gradient(q_loss2, self.policy_with_value.Q2.trainable_weights)
            q_gradient_cost = tape.gradient(q_loss_cost, self.policy_with_value.Q_cost.trainable_weights)

        return q_loss1, q_loss2, q_loss_cost, q_gradient1, q_gradient2, q_gradient_cost

    @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape() as tape:
            processed_obses = self.preprocessor.tf_process_obses(mb_obs)
            actions, logps = self.policy_with_value.compute_action(processed_obses)
            all_Qs1 = self.policy_with_value.compute_Q1(processed_obses, actions)
            all_Qs2 = self.policy_with_value.compute_Q2(processed_obses, actions)
            all_Qs_min = self.tf.reduce_min((all_Qs1, all_Qs2), 0)
            alpha = self.tf.exp(self.policy_with_value.log_alpha) if self.args.alpha == 'auto' else self.args.alpha
            policy_loss = self.tf.reduce_mean(alpha*logps-all_Qs_min)

            policy_entropy = -self.tf.reduce_mean(logps)
            value_var = self.tf.math.reduce_variance(all_Qs_min)
            value_mean = self.tf.reduce_mean(all_Qs_min)

        with self.tf.name_scope('policy_gradient') as scope:
            policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights,)
            return policy_loss, policy_gradient, policy_entropy, value_mean, value_var

    @tf.function
    def alpha_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape() as tape:
            processed_obses = self.preprocessor.tf_process_obses(mb_obs)
            actions, logps = self.policy_with_value.compute_action(processed_obses)
            log_alpha = self.policy_with_value.log_alpha
            alpha_loss = self.tf.reduce_mean(-log_alpha * self.tf.stop_gradient(logps + self.args.target_entropy))

        with self.tf.name_scope('alpha_gradient') as scope:
            alpha_gradient = tape.gradient(alpha_loss, self.policy_with_value.alpha_model.trainable_weights)
            return alpha_loss, self.tf.exp(log_alpha), alpha_gradient

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        mb_targets = self.batch_data['batch_targets']
        mb_actions = self.batch_data['batch_actions']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions, mb_targets)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.alpha_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="alpha_forward_and_backward", step=0)

    def compute_gradient(self, batch_data, rb, indexes, iteration):  # compute gradient
        if self.counter % self.num_batch_reuse == 0:
            self.get_batch_data(batch_data, rb, indexes)
        self.counter += 1
        if self.args.buffer_type != 'normal':
            self.info_for_buffer.update(dict(td_error=self.compute_td_error()))
        mb_obs = self.batch_data['batch_obs']
        mb_actions = self.batch_data['batch_actions']
        mb_targets = self.batch_data['batch_targets']
        mb_cost_targets = self.batch_data['batch_costs']

        with self.q_gradient_timer:
            q_loss1, q_loss2, q_gradient1, q_gradient2 = self.q_forward_and_backward(mb_obs, mb_actions, mb_targets, mb_cost_targets)
            q_gradient1, q_gradient_norm1 = self.tf.clip_by_global_norm(q_gradient1, self.args.gradient_clip_norm)
            q_gradient2, q_gradient_norm2 = self.tf.clip_by_global_norm(q_gradient2, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            policy_loss, policy_gradient, policy_entropy, value_mean, value_var = self.policy_forward_and_backward(mb_obs)

        policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                            self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            target_time=self.target_timer.mean,
            q_loss1=q_loss1.numpy(),
            q_loss2=q_loss2.numpy(),
            policy_loss=policy_loss.numpy(),
            policy_entropy=policy_entropy.numpy(),
            mb_targets_mean=np.mean(mb_targets),
            value_mean=value_mean.numpy(),
            value_var=value_var.numpy(),
            q_gradient_norm1=q_gradient_norm1.numpy(),
            q_gradient_norm2=q_gradient_norm2.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
        ))
        if self.args.alpha == 'auto':
            with self.alpha_timer:
                alpha_loss, alpha, alpha_gradient = self.alpha_forward_and_backward(mb_obs)
                alpha_gradient, alpha_gradient_norm = self.tf.clip_by_global_norm(alpha_gradient,
                                                                                  self.args.gradient_clip_norm)
            self.stats.update(dict(alpha=alpha.numpy(),
                                   alpha_loss=alpha_loss.numpy(),
                                   alpha_gradient_norm=alpha_gradient_norm.numpy(),
                                   alpha_time=self.alpha_timer.mean))

            gradient_tensor = q_gradient1 + q_gradient2 + policy_gradient + alpha_gradient
        else:
            gradient_tensor = q_gradient1 + q_gradient2 + policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))


if __name__ == '__main__':
    pass
