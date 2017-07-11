import numpy as np
import tensorflow as tf


class Policy(object):
    def __init__(self, obs_dim, act_dim, hid_units=30):
        self._build_graph(obs_dim, act_dim, hid_units)
        self._init_session()

    def _build_graph(self, obs_dim, act_dim, hid_units):
        """ Build TensorFlow graph"""
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders(obs_dim, act_dim)
            self._policy_nn(hid_units, obs_dim, act_dim)
            self._logprob(act_dim)
            self._kl_entropy(act_dim)
            self._sample(act_dim)
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self, obs_dim, act_dim):
        """ Input placeholders"""
        self.obs_ph = tf.placeholder(tf.float32, (None, obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        self.training_ph = tf.placeholder(tf.bool, (), 'training')
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        self.old_log_vars_ph = tf.placeholder(tf.float32, (act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, act_dim), 'old_means')

    def _policy_nn(self, hid_units, obs_dim, act_dim):
        """ Neural net for policy approximation function """
        # normed = tf.layers.batch_normalization(self.obs_ph, training=self.training_ph)
        # hidden layers
        out = tf.layers.dense(self.obs_ph, 64, tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(2 / obs_dim)),
                              name="h1")
        out = tf.layers.dense(out, 64, tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(2 / 64)),
                              name="h2")
        # outputs
        self.means = tf.layers.dense(out, act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(2 / 64)),
                                     name="mu")
        log_var_10 = tf.get_variable("logvar", [2, act_dim], initializer=tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_var_10, axis=0)
        # self.log_vars = tf.get_variable("log_vars", (act_dim,),
        #                                 initializer=tf.constant_initializer(0.0))

    def _logprob(self, act_dim):
        """ Log probabilities of batch of states, actions"""
        logp = -0.5 * (np.log(np.sqrt(2.0 * np.pi)) * act_dim)
        logp += -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

    def _kl_entropy(self, act_dim):
        """
        Add KL divergence between old and new distributions
        Add entropy of present policy given states and actions
        """
        det_cov_old = tf.exp(tf.reduce_sum(self.old_log_vars_ph))
        det_cov_new = tf.exp(tf.reduce_sum(self.log_vars))
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * (tf.log(det_cov_new) - tf.log(det_cov_old) + tr_old_new +
                         tf.reduce_mean(tf.square(self.means - self.old_means_ph) /
                                        tf.exp(self.log_vars)) - act_dim)
        self.entropy = 0.5 * (act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_mean(self.log_vars))

    def _sample(self, act_dim):
        """ Sample from distribution, given observation"""
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(act_dim,)))

    def _loss_train_op(self, mom=0.9):
        """

        Args:
            mom:

        Returns:

        """
        self.loss = -tf.reduce_mean(self.advantages_ph * self.logp)
        # beta_ph: hyper-parameter to control weight of kl-divergence loss
        # self.loss += -tf.reduce_mean(self.beta_ph * self.kl)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # optimizer = tf.train.MomentumOptimizer(self.lr_ph, mom)
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        # with tf.control_dependencies(update_ops):
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs,
                     self.training_ph: False}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, epochs=1, beta=0.0001, lr=1e-4):
        """Perform policy update based on batch (size = N) of samples

        Args:
            lr:
            observes:
            actions:
            advantages:
            epochs:
            beta:

        Returns:

        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: beta,
                     self.training_ph: False,
                     self.lr_ph: lr}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        for e in range(epochs):
            feed_dict[self.training_ph] = True
            feed_dict[self.old_log_vars_ph] = old_log_vars_np
            feed_dict[self.old_means_ph] = old_means_np
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)

        feed_dict[self.training_ph] = False
        loss, entropy, kl = self.sess.run([self.loss, self.entropy, self.kl], feed_dict)

        metrics = {'AvgLoss': loss,
                   'AvgEntropy': entropy,
                   'OldNewKL': kl}

        return metrics

    def close_sess(self):
        self.sess.close()
