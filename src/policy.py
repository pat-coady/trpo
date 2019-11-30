"""
NN Policy with KL Divergence Constraint (PPO / TRPO)

Written by Patrick Coady (pat-coady.github.io)
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam

import numpy as np


class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, init_logvar, clipping_range=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            init_logvar: natural log of initial policy variance
        """
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult
        self.init_logvar = init_logvar
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clipping_range = clipping_range
        self.policy = self._build_policy_model()
        self.logprob = self._build_logprob_model()
        self.kl_entropy = self._build_kl_entropy_model()
        self.train = self.__build_train_model()

    def _build_policy_model(self):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_units = self.obs_dim * self.hid1_mult
        hid3_units = self.act_dim * 10  # 10 empirically determined
        hid2_units = int(np.sqrt(hid1_units * hid3_units))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(hid2_units)  # 9e-4 empirically determined
        obs = Input(shape=(self.obs_dim,), dtype='float32')
        y = Dense(hid1_units, activation='tanh')(obs)
        y = Dense(hid2_units, activation='tanh')(y)
        y = Dense(hid3_units, activation='tanh')(y)
        act_means = Dense(self.act_dim)(y)
        # logvar_speed increases learning rate for log-variances.
        # heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_units) // 48
        act_logvars = K.variable(np.zeros(size=(self.act_dim, logvar_speed)), dtype='float32')
        act_logvars = K.sum(act_logvars, axis=-1) + self.init_logvar
        act_means_dummy = 0 * act_means  # dummy graph connection for act_logvars
        act_logvars = act_logvars + act_means_dummy
        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_units, hid2_units, hid3_units, self.lr, logvar_speed))

        return Model(inputs=obs, outputs=[act_means, act_logvars])

    def _build_logprob_model(self):
        """ Model calculates log probabilities of a batch of actions."""
        actions = Input(shape=(self.act_dim,), dtype='float32')
        act_means = Input(shape=(self.act_dim,), dtype='float32')
        act_logvars = Input(shape=(self.act_dim,), dtype='float32')

        logp = -0.5 * K.sum(act_logvars)
        logp += -0.5 * K.sum(K.square(actions - act_means) / K.exp(act_logvars))

        return Model(inputs=[actions, act_means, act_logvars], outputs=logp)

    def _build_kl_entropy_model(self):
        """
        Model calculates:
            1. KL divergence between old and new distributions
            2. Entropy of present policy

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        old_means = Input(shape=(self.act_dim,), dtype='float32')
        old_logvars = Input(shape=(self.act_dim,), dtype='float32')
        new_means = Input(shape=(self.act_dim,), dtype='float32')
        new_logvars = Input(shape=(self.act_dim,), dtype='float32')
        log_det_cov_old = K.sum(old_logvars)
        log_det_cov_new = K.sum(new_logvars)
        trace_old_new = tf.reduce_sum(tf.exp(old_logvars - new_logvars))

        kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + trace_old_new +
                                  K.sum(K.square(new_means - old_means) /
                                        K.exp(new_logvars)) -
                                  self.act_dim)
        entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                         K.sum(new_logvars))

        return Model(inputs=[old_means, old_logvars, new_means, new_logvars],
                     outputs=[kl, entropy])

    def __build_train_model(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """
        old_means = Input(shape=(self.act_dim,), dtype='float32')
        old_logvars = Input(shape=(self.act_dim,), dtype='float32')
        logp_old = Input(shape=(1,), dtype='float32')
        observes = Input(shape=(self.obs_dim,), dtype='float32')
        new_means, new_logvars = self.policy(observes)
        actions = Input(shape=(self.act_dim,), dtype='float32')
        logp_new = self.logprob([actions, new_means, new_logvars])
        advantages = Input(shape=(1,), dtype='float32')
        kl, entropy = self.kl_entropy([old_means, old_logvars,
                                       new_means, new_logvars])

        if self.clipping_range is not None:
            print('setting up loss with clipping objective')
            pg_ratio = K.exp(logp_new - logp_old)
            clipped_pg_ratio = K.clip(pg_ratio, 1 - self.clipping_range[0],
                                      1 + self.clipping_range[1])
            surrogate_loss = K.minimum(advantages * pg_ratio,
                                       advantages * clipped_pg_ratio)
            loss = -tf.reduce_mean(surrogate_loss)
        else:
            print('setting up loss with KL penalty')
            beta = K.variable(self.beta, dtype='float32', name='beta')
            eta = K.variable(self.beta, dtype='float32', name='eta')
            loss1 = -tf.reduce_mean(advantages *
                                    tf.exp(logp_new - logp_old))
            loss2 = tf.reduce_mean(beta * kl)
            loss3 = eta * tf.square(tf.maximum(0.0, kl - 2.0 * self.kl_targ))
            loss = loss1 + loss2 + loss3
        loss = Lambda(lambda x: x, name='loss')(loss)
        kl = Lambda(lambda x: x, name='kl')(kl)
        entropy = Lambda(lambda x: x, name='entropy')(entropy)
        model = Model(inputs=[observes, actions, advantages,
                              logp_old, old_means, old_logvars],
                      outputs=[loss, kl, entropy])
        optimizer = Adam(self.lr * self.lr_multiplier)
        model.compile(optimizer=optimizer, loss={'loss': 'mae'})

        return model

    def sample(self, obs):
        """Draw sample from policy."""
        act_means, act_logvars = self.policy.predict(obs)
        act_stddevs = np.exp(act_logvars.numpy() / 2)

        return np.random.normal(act_means.numpy(), act_stddevs)

    def update(self, observes, actions, advantages, logger):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        K.set_value(self.train.optimizer.lr, self.lr * self.lr_multiplier)
        K.set_value(self.train.beta, self.beta)
        K.set_value(self.train.eta, self.eta)
        old_means, old_logvars = self.policy.predict(observes)
        old_logp = self.logprob.predict(actions, old_means, old_logvars)
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            loss, kl, entropy = self.train([observes, actions, advantages,
                                            old_logp, old_means, old_logvars])
            kl = np.mean(kl)
            entropy = np.mean(entropy)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})
