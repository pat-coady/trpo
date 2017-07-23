"""
Various Procedures Not Used in Final Implementation

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np


class ConstantScaler(object):
    """ Dumb scaler, scale and offset set at initialization """
    def __init__(self, obs_dim, scale=1.0, offset=0.0):
        self.scale = np.ones(obs_dim) * scale
        self.offset = np.zeros(obs_dim) + offset

    def update(self, x):
        pass  # no updates for constant scaler

    def get(self):
        """ resturns 2-tuple: (scale, offset) """
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
    """ Adds estimated advantage to all timesteps of all trajectories

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        trajectory['advantages'] = trajectory['disc_sum_rew'] - trajectory['values']
