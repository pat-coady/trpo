import numpy as np
import tensorflow as tf


class Policy(object):
    def __init__(self, obs_dim, act_dim, kl_targ=0.003):
        self.beta = 1.0
        self.kl_targ = kl_targ
        self._build_graph(obs_dim, act_dim)
        self._init_session()

    def _build_graph(self, obs_dim, act_dim):
        """ Build TensorFlow graph"""
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders(obs_dim, act_dim)
            self._policy_nn(obs_dim, act_dim)
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
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        self.old_log_vars_ph = tf.placeholder(tf.float32, (act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, act_dim), 'old_means')

    def _policy_nn(self, obs_dim, act_dim):
        """ Neural net for policy approximation function """
        out = tf.layers.dense(self.obs_ph, 30, tf.nn.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(2 / obs_dim)),
                              name="h1")
        self.means = tf.layers.dense(out, act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(2 / 30)),
                                     name="mu")
        self.log_vars = tf.get_variable("log_vars", (act_dim,),
                                        initializer=tf.constant_initializer(0.0))

    def _logprob(self, act_dim):
        """ Log probabilities of batch of states, actions"""
        logp = -0.5 * (np.log(np.sqrt(2.0 * np.pi)) * act_dim)
        logp += -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * (np.log(np.sqrt(2.0 * np.pi)) * act_dim)
        logp_old += -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

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

    def _loss_train_op(self):
        self.loss = -tf.reduce_mean(self.advantages_ph *
                                    tf.exp(self.logp - self.logp_old))
        self.loss += tf.reduce_mean(self.beta_ph * self.kl)
        optimizer = tf.train.AdamOptimizer(0.003)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, epochs=50):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        for e in range(epochs):
            feed_dict[self.old_log_vars_ph] = old_log_vars_np
            feed_dict[self.old_means_ph] = old_means_np
            _, loss, kl = self.sess.run([self.train_op, self.loss, self.kl], feed_dict)
            if kl > self.kl_targ * 4:
                break
        if kl > self.kl_targ * 2:
            self.beta *= 1.5
        elif kl < self.kl_targ / 2:
            self.beta /= 1.5


        loss, entropy, kl = self.sess.run([self.loss, self.entropy, self.kl], feed_dict)

        metrics = {'AvgLoss': loss,
                   'AvgEntropy': entropy,
                   'OldNewKL': kl}

        return metrics

    def close_sess(self):
        self.sess.close()
