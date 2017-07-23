"""
Archive of Procedures Not Used in Final Implementation

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import tensorflow as tf


class ConstantScaler(object):
    """ Dumb scaler, scale and offset set at initialization """
    def __init__(self, obs_dim, scale=1.0, offset=0.0):
        self.scale = np.ones(obs_dim) * scale
        self.offset = np.zeros(obs_dim) + offset

    def update(self, x):
        pass  # no updates for constant scaler

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return self.scale, self.offset


class LinearValueFunction(object):
    """Simple linear regression value function, uses linear and squared features.

    Mostly copied from: https://github.com/joschu/modular_rl
    """
    def __init__(self):
        self.coef = None

    def fit(self, x, y, logger):
        """ Fit model - (i.e. solve normal equations)

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        y_hat = self.predict(x)
        old_exp_var = 1-np.var(y-y_hat)/np.var(y)
        xp = self.preproc(x)
        a = xp.T.dot(xp)
        nfeats = xp.shape[1]
        a[np.arange(nfeats), np.arange(nfeats)] += 1e-3  # a little ridge regression
        b = xp.T.dot(y)
        self.coef = np.linalg.solve(a, b)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat-y))
        exp_var = 1-np.var(y-y_hat)/np.var(y)

        logger.log({'LinValFuncLoss': loss,
                    'LinExplainedVarNew': exp_var,
                    'LinExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Predict method, predict zeros if model untrained """
        if self.coef is None:
            return np.zeros(x.shape[0])
        else:
            return self.preproc(x).dot(self.coef)

    @staticmethod
    def preproc(X):
        """ Adds squared features and bias term """

        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


def add_advantage(trajectories):
    """ Adds estimated advantage to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        trajectory['advantages'] = trajectory['disc_sum_rew'] - trajectory['values']


class PolicyWithVariance(object):
    """ Neural Net output means AND variance (had poor performance) """
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
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        self.old_log_vars_ph = tf.placeholder(tf.float32, (None, act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, act_dim), 'old_means')

    def _policy_nn(self, obs_dim, act_dim):
        """ Neural net for policy approximation function """
        out = tf.layers.dense(self.obs_ph, 200, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / obs_dim)),
                              name="h1")
        out = tf.layers.dense(out, 100, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / 200)),
                              name="h2")
        out = tf.layers.dense(out, 50, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / 100)),
                              name="h3")
        self.means = tf.layers.dense(out, act_dim,
                                     kernel_initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(1 / 50)),
                                     name="means")
        self.log_vars = tf.layers.dense(out, act_dim,
                                        kernel_initializer=tf.random_normal_initializer(
                                            stddev=np.sqrt(1 / 50)),
                                        name="log_vars")

    def _logprob(self, act_dim):
        """ Log probabilities of batch of states, actions"""
        logp = -0.5 * (np.log(np.sqrt(2.0 * np.pi)) * act_dim)
        # logp += -0.5 * tf.reduce_sum(self.log_vars, axis=1)
        logp += -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * (np.log(np.sqrt(2.0 * np.pi)) * act_dim)
        logp_old += -0.5 * tf.reduce_sum(self.old_log_vars_ph, axis=1)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

    def _kl_entropy(self, act_dim):
        """
        Add KL divergence between old and new distributions
        Add entropy of present policy given states and actions
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph, axis=1)
        log_det_cov_new = tf.reduce_sum(self.log_vars, axis=1)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars), axis=1)

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) - act_dim)

        self.entropy = 0.5 * (act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_mean(tf.reduce_sum(self.log_vars, axis=1)))

    def _sample(self, act_dim):
        """ Sample from distribution, given observation"""
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(act_dim,)))

    def _loss_train_op(self):
        # TODO: use reduce_mean or reduce_sum?
        loss1 = -tf.reduce_mean(self.advantages_ph *
                                tf.exp(self.logp - self.logp_old))
        loss2 = tf.reduce_mean(self.beta_ph * self.kl)
        loss3 = self.eta_ph * tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
        self.loss = loss1 + loss2 + loss3
        # optimizer = tf.train.AdamOptimizer(0.00003)
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger, epochs=20):
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: 100}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        for e in range(epochs):
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:
                break
        if kl > self.kl_targ * 2:
            self.beta *= 1.5
        elif kl < self.kl_targ / 2:
            self.beta /= 1.5

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta})

    def close_sess(self):
        self.sess.close()