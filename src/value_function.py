import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class ValueFunction(object):

    def __init__(self, obs_dim, epochs=5, reg=1e-2, lr=1e-2):
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
            self.obs = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
            self.val = tf.placeholder(dtype=tf.float32, shape=(None,))
            out = tf.layers.dense(self.obs, 64, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(2 / self.obs_dim)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            out = tf.layers.dense(out, 32, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(2 / 64)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(2 / 32)),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val))
            self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * self.reg
            optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y):
        batch_size = 64
        # self.sess.run(self.init)
        for e in range(self.epochs):
            X, y = shuffle(x, y)
            for j in range(x.shape[0] // batch_size):
                start = j*batch_size
                end = (j+1)*batch_size
                feed_dict = {self.obs: X[start:end, :],
                             self.val: y[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat-y))

        return loss

    def predict(self, x):
        feed_dict = {self.obs: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        self.sess.close()
