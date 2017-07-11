import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class LinearValueFunction(object):
    coef = None

    def fit(self, x, y):
        xp = self.preproc(x)
        a = xp.T.dot(xp)
        nfeats = xp.shape[1]
        a[np.arange(nfeats), np.arange(nfeats)] += 1e-3  # a little ridge regression
        b = xp.T.dot(y)
        self.coef = np.linalg.solve(a, b)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat-y))
        exp_var = 1-np.var(y-y_hat)/np.var(y)

        return {'ValFuncLoss': loss,
                'ExplainedVar': exp_var}

    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)

    def preproc(self, X):

        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


class ValueFunction(object):

    def __init__(self, obs_dim, epochs=10, reg=1e-4, lr=1e-4):
        self.obs_dim = obs_dim
        self.epochs = epochs
        self.reg = reg
        self.lr = lr
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            out = tf.layers.dense(self.obs, 128, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(2/self.obs_dim)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            out = tf.layers.dense(out, 64, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(2/128)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            out = tf.layers.dense(out, 32, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(2/64)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            self.out = tf.layers.dense(out, 1,
                                       kernel_initializer=tf.random_normal_initializer(
                                           stddev=np.sqrt(2/32)),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            self.loss = tf.reduce_mean(tf.square(self.out - self.val))
            # self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * self.reg
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=False)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y):
        batch_size = 64
        for e in range(self.epochs):
            x, y = shuffle(x, y)
            for j in range(x.shape[0] // batch_size):
                start = j*batch_size
                end = (j+1)*batch_size
                feed_dict = {self.obs: x[start:end, :],
                             self.val: y[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat-y))
        exp_var = 1-np.var(y-y_hat)/np.var(y)

        return {'ValFuncLoss': loss,
                'ExplainedVar': exp_var}

    def predict(self, x):
        feed_dict = {self.obs: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        self.sess.close()
