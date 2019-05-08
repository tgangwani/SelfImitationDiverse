import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from imitation.common import tf_util as U
from imitation.common.running_mean_std import TfRunningMeanStd
from imitation.common.misc_util import logit_bernoulli_entropy

class Discriminator():
  def __init__(self, env, hidden_size, entcoeff, scope="discriminator"):
    self.scope = scope
    self.observation_shape = env.observation_space.shape
    self.action_shape = env.action_space.shape
    self.hidden_size = hidden_size

    assert len(self.observation_shape) == len(self.action_shape) == 1, "Current implementation supports flat obs/acs spaces."
    self.input_dim = self.observation_shape[0] + self.action_shape[0]

    # Create placeholders
    self.build_ph()

    # Build grpah
    policy_logits = self.build_graph(self.policy_obs_ph, self.policy_acs_ph, reuse=False)
    expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)

    # Build accuracy
    policy_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(policy_logits) < 0.5))
    expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))

    policy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=policy_logits, labels=tf.zeros_like(policy_logits))
    policy_loss = tf.reduce_mean(policy_loss)

    expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
    expert_loss = tf.reduce_sum(self.expert_wts_ph * expert_loss)

    # Build entropy loss
    logits = tf.concat([policy_logits, expert_logits], 0)
    entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
    entropy_loss = -entcoeff*entropy

    # Loss + Accuracy terms
    self.losses = [policy_loss, expert_loss, entropy, entropy_loss, policy_acc, expert_acc]
    self.loss_names = ["policy_loss", "expert_loss", "entropy", "entropy_loss", "policy_acc", "expert_acc"]
    self.total_loss = policy_loss + expert_loss + entropy_loss

    # Build Reward for policy
    self.reward_op = -tf.log(1-tf.nn.sigmoid(policy_logits) + 1e-8)

    var_list = self.get_trainable_variables()
    self.lossandgrad = U.function([self.policy_obs_ph, self.policy_acs_ph, self.expert_obs_ph, self.expert_acs_ph, self.expert_wts_ph],
                         self.losses + [U.flatgrad(self.total_loss, var_list)])

  def build_ph(self):
    self.policy_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="policy_obs_ph")
    self.policy_acs_ph = tf.placeholder(tf.float32, (None, ) + self.action_shape, name="policy_acs_ph")
    self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_obs_ph")
    self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.action_shape, name="expert_acs_ph")
    self.expert_wts_ph = tf.placeholder(tf.float32, (None, 1), name='expert_wts_ph')

  def build_graph(self, obs_ph, acs_ph, reuse=False):
    with tf.variable_scope(self.scope):
      if reuse:
        tf.get_variable_scope().reuse_variables()

      with tf.variable_scope("inputfilter"):
          self.in_rms = TfRunningMeanStd(shape=self.input_dim)

      x = tf.concat([obs_ph, acs_ph], axis=1)
      x = tf.clip_by_value((x - self.in_rms.mean) / self.in_rms.std, -5.0, 5.0)
      x = tc.layers.fully_connected(x, self.hidden_size, activation_fn=tf.nn.tanh)
      x = tc.layers.fully_connected(x, self.hidden_size, activation_fn=tf.nn.tanh)
      logits = tc.layers.fully_connected(x, 1, activation_fn=tf.identity)

    return logits

  def get_trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

  def get_reward(self, obs, acs):
    sess = U.get_session()
    if len(obs.shape) == 1:
      obs = np.expand_dims(obs, 0)
    if len(acs.shape) == 1:
      acs = np.expand_dims(acs, 0)
    feed_dict = {self.policy_obs_ph:obs, self.policy_acs_ph:acs}
    return sess.run(self.reward_op, feed_dict)
