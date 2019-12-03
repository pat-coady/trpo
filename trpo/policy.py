"""
NN Policy with KL Divergence Constraint

Written by Patrick Coady (pat-coady.github.io)
"""
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.optimizers import Adam
import numpy as np


class Policy(object):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, init_logvar):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            init_logvar: natural log of initial policy variance
        """
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = kl_targ
        self.epochs = 20
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.trpo = TRPO(obs_dim, act_dim, hid1_mult, kl_targ, init_logvar, eta)
        self.policy = self.trpo.get_layer('policy_nn')
        self.lr = self.policy.get_lr()  # lr calculated based on size of PolicyNN
        self.trpo.compile(optimizer=Adam(self.lr * self.lr_multiplier))
        self.logprob_calc = LogProb()

    def sample(self, obs):
        """Draw sample from policy."""
        act_means, act_logvars = self.policy(obs)
        act_stddevs = np.exp(act_logvars / 2)

        return np.random.normal(act_means, act_stddevs).astype(np.float32)

    def update(self, observes, actions, advantages, logger):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
            logger: Logger object, see utils.py
        """
        K.set_value(self.trpo.optimizer.lr, self.lr * self.lr_multiplier)
        K.set_value(self.trpo.beta, self.beta)
        old_means, old_logvars = self.policy(observes)
        old_means = old_means.numpy()
        old_logvars = old_logvars.numpy()
        old_logp = self.logprob_calc([actions, old_means, old_logvars])
        old_logp = old_logp.numpy()
        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            loss = self.trpo.train_on_batch([observes, actions, advantages,
                                             old_means, old_logvars, old_logp])
            kl, entropy = self.trpo.predict_on_batch([observes, actions, advantages,
                                                      old_means, old_logvars, old_logp])
            kl, entropy = np.mean(kl), np.mean(entropy)
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


class PolicyNN(Layer):
    """ Neural net for policy approximation function.

    Policy parameterized by Gaussian means and variances. NN outputs mean
     action based on observation. Trainable variables hold log-variances
     for each action dimension (i.e. variances not determined by NN).
    """
    def __init__(self, obs_dim, act_dim, hid1_mult, init_logvar, **kwargs):
        super(PolicyNN, self).__init__(**kwargs)
        self.batch_sz = None
        self.init_logvar = init_logvar
        hid1_units = obs_dim * hid1_mult
        hid3_units = act_dim * 10  # 10 empirically determined
        hid2_units = int(np.sqrt(hid1_units * hid3_units))
        self.lr = 9e-4 / np.sqrt(hid2_units)  # 9e-4 empirically determined
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.dense1 = Dense(hid1_units, activation='tanh', input_shape=(obs_dim,))
        self.dense2 = Dense(hid2_units, activation='tanh', input_shape=(hid1_units,))
        self.dense3 = Dense(hid3_units, activation='tanh', input_shape=(hid2_units,))
        self.dense4 = Dense(act_dim, input_shape=(hid3_units,))
        # logvar_speed increases learning rate for log-variances.
        # heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_units) // 48
        self.logvars = self.add_weight(shape=(logvar_speed, act_dim),
                                       trainable=True, initializer='zeros')
        print('Policy Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}, logvar_speed: {}'
              .format(hid1_units, hid2_units, hid3_units, self.lr, logvar_speed))

    def build(self, input_shape):
        self.batch_sz = input_shape[0]

    def call(self, inputs, **kwargs):
        y = self.dense1(inputs)
        y = self.dense2(y)
        y = self.dense3(y)
        means = self.dense4(y)
        logvars = K.sum(self.logvars, axis=0, keepdims=True) + self.init_logvar
        logvars = K.tile(logvars, (self.batch_sz, 1))

        return [means, logvars]

    def get_lr(self):
        return self.lr


class KLEntropy(Layer):
    """
    Layer calculates:
        1. KL divergence between old and new distributions
        2. Entropy of present policy

    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
    """
    def __init__(self, **kwargs):
        super(KLEntropy, self).__init__(**kwargs)
        self.act_dim = None

    def build(self, input_shape):
        self.act_dim = input_shape[0][1]

    def call(self, inputs, **kwargs):
        old_means, old_logvars, new_means, new_logvars = inputs
        log_det_cov_old = K.sum(old_logvars, axis=-1, keepdims=True)
        log_det_cov_new = K.sum(new_logvars, axis=-1, keepdims=True)
        trace_old_new = K.sum(K.exp(old_logvars - new_logvars), axis=-1, keepdims=True)
        kl = 0.5 * (log_det_cov_new - log_det_cov_old + trace_old_new +
                    K.sum(K.square(new_means - old_means) /
                          K.exp(new_logvars), axis=-1, keepdims=True) -
                    np.float32(self.act_dim))
        entropy = 0.5 * (np.float32(self.act_dim) * (np.log(2 * np.pi) + 1.0) +
                         K.sum(new_logvars, axis=-1, keepdims=True))

        return [kl, entropy]


class LogProb(Layer):
    """Layer calculates log probabilities of a batch of actions."""
    def __init__(self, **kwargs):
        super(LogProb, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        actions, act_means, act_logvars = inputs
        logp = -0.5 * K.sum(act_logvars, axis=-1, keepdims=True)
        logp += -0.5 * K.sum(K.square(actions - act_means) / K.exp(act_logvars),
                             axis=-1, keepdims=True)

        return logp


class TRPO(Model):
    def __init__(self, obs_dim, act_dim, hid1_mult, kl_targ, init_logvar, eta, **kwargs):
        super(TRPO, self).__init__(**kwargs)
        self.kl_targ = kl_targ
        self.eta = eta
        self.beta = self.add_weight('beta', initializer='zeros', trainable=False)
        self.policy = PolicyNN(obs_dim, act_dim, hid1_mult, init_logvar)
        self.logprob = LogProb()
        self.kl_entropy = KLEntropy()

    def call(self, inputs):
        obs, act, adv, old_means, old_logvars, old_logp = inputs
        new_means, new_logvars = self.policy(obs)
        new_logp = self.logprob([act, new_means, new_logvars])
        kl, entropy = self.kl_entropy([old_means, old_logvars,
                                       new_means, new_logvars])
        loss1 = -K.mean(adv * K.exp(new_logp - old_logp))
        loss2 = K.mean(self.beta * kl)
        # TODO - Take mean before or after hinge loss?
        loss3 = self.eta * K.square(K.maximum(0.0, K.mean(kl) - 2.0 * self.kl_targ))
        self.add_loss(loss1 + loss2 + loss3)

        return [kl, entropy]
