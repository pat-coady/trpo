import gym
import numpy as np
import tensorflow as tf


class Policy(object):
    def __init__(self, obs_dim, action_dim):
        self._build_graph()
        self._init_session()

    def sample(self, obs):
        """ Draw sample from policy distribution"""
        pass

    def _build_graph(self):
        """ Build graph with policy, d_kl and loss"""
        pass

    def _init_session(self):
        """ Launch TensorFlow session and initialize variables"""

    def update(self, observes, actions, advantages):
        """ Perform policy update based on batch of samples

        return loss and d_kl
        """
        pass

    def close_sess(self):
        pass


class ValueFunction(object):

    def __init__(self, obs_dim, epochs=5, reg=1e-2, lr=1e-2):
        self._build_graph()
        self._init_sess()

    def fit(self, observes, disc_sum_rew):
        pass

    def predict(self, observes):
        pass

    def close_sess(self):
        pass


def init_gym(env_name='Pendulum-v0'):
    """

    :param env_name: str, OpenAI Gym environment name
    :return: 3-tuple
        env: ai gym environment
        obs_dim: observation dimensions
        act_dim: action dimensions
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_policy(env, policy, min_steps):
    """ Run policy and collect data for a minimum of min_steps

    :param env: ai gym environment
    :param policy: policy with "sample" method
    :param min_steps: minimum number of samples to collect, completes current
    episode after min_steps reached
    :return: list dictionaries, 1 dictionary per episode. Dict key/values:
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (undiscounted) rewards from episode
    """


def add_disc_sum_rew(trajectories, gamma=1.0):
    """ Adds discounted sum of rewards to all timesteps of all trajectories

    :param trajectories: as returned by run_policy()
    :return: None (mutates trajectories to add 'disc_sum_rew' key)
    """


def add_value(trajectories, val_func):
    """ Adds estimated value to all timesteps of all trajectories

    :param trajectories: as returned by run_policy()
    :return: None (mutates trajectories to add 'value' key)
    """

def add_advantage(trajectories, val_func):
    """ Adds estimated advantage to all timesteps of all trajectories

    :param trajectories: as returned by run_policy()
    :return: None (mutates trajectories to add 'advantage' key)
    """

def build_train_set(trajectories):
    """ Concatenates all trajectories into single NumPy array with first
     dimension = N = total time steps across all trajectories

    :param trajectories: trajectories after processing by add_disc_sum_rew(),
     add_value(), add_advantage()
    :return: 4-tuple of NumPy arrays
    obs: shape = (N, obs_dim)
    actions: shape = (N, act_dim)
    advantages: shape = (N,)
    disc_sum_rew: shape = (N,)
    """


def main(num_iter=100,
         gamma=1.0):

    # launch ai gym env
    env, obs_dim, act_dim = init_gym()

    # init value function and policy
    val_func = ValueFunction(obs_dim)
    policy = Policy(obs_dim, act_dim)
    # main loop:
    for i in range(num_iter):
        # collect data batch using policy
        trajectories = run_policy(env, policy)
        #   calculate cum_sum_rew: all time steps
        add_disc_sum_rew(trajectories, gamma)
        #   value prediction: all time steps
        add_value(trajectories, val_func)
        #   calculate advantages: cum_sum_rew - v(s_t)
        add_advantage(trajectories)
        #   policy update
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        policy.update(observes, actions, advantages)
        #   fit value function
        val_func.fit(observes, disc_sum_rew)

    pass








policy = Policy(obs_dim, action_dim)