"""
State-Value Function

Written by Patrick Coady (pat-coady.github.io)
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np


class NNValueFunction(object):
    """ NN-based state-value function """
    def __init__(self, obs_dim, hid1_mult):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.epochs = 10
        self.lr = None  # learning rate set in _build_model()
        self.model = self._build_model()

    def _build_model(self):
        """ Construct TensorFlow graph, including loss function, init op and train op """
        obs = Input(shape=(self.obs_dim,), dtype='float32')
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        hid1_units = self.obs_dim * self.hid1_mult
        hid3_units = 5  # 5 chosen empirically on 'Hopper-v1'
        hid2_units = int(np.sqrt(hid1_units * hid3_units))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 1e-2 / np.sqrt(hid2_units)  # 1e-2 empirically determined
        print('Value Params -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
              .format(hid1_units, hid2_units, hid3_units, self.lr))
        y = Dense(hid1_units, activation='tanh')(obs)
        y = Dense(hid2_units, activation='tanh')(y)
        y = Dense(hid3_units, activation='tanh')(y)
        y = Dense(1)(y)
        model = Model(inputs=obs, outputs=y)
        optimizer = Adam(self.lr)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def fit(self, x, y, logger):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.model.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=batch_size,
                       shuffle=True, verbose=0)
        y_hat = self.model.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

        logger.log({'ValFuncLoss': loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

    def predict(self, x):
        """ Predict method """
        return self.model.predict(x)
