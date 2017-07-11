import gym
from policy import *
from value_function import *
import scipy.signal


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


def run_episode(env, policy, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy with "sample" method
        animate: boolean, True uses env.render() method to animate episode

    Returns: 3-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
    """
    obs = env.reset()
    observes, actions, rewards = [], [], []
    done = False
    while not done:
        if animate:
            env.render()
        obs = obs.reshape((1, -1))
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1))
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

    return np.concatenate(observes), np.concatenate(actions), np.concatenate(rewards)


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
    total_steps = 0
    trajectories = []
    while total_steps < min_steps:
        observes, actions, rewards = run_episode(env, policy)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards}
        trajectories.append(trajectory)

    return trajectories


def view_policy(env, policy):
    """ Run policy and view using env.render() method

    :param env: ai gym environment
    :param policy: policy with "sample" method
    :return: None
    """
    return run_episode(env, policy, animate=True)


def add_disc_sum_rew(trajectories, gamma=1.0):
    """ Adds discounted sum of rewards to all timesteps of all trajectories

    :param trajectories: as returned by run_policy()
    :return: None (mutates trajectories to add 'disc_sum_rew' key)
    """
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        disc_sum_rew = scipy.signal.lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all timesteps of all trajectories

    :param trajectories: as returned by run_policy()
    :return: None (mutates trajectories to add 'values' key)
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_advantage(trajectories):
    """ Adds estimated advantage to all timesteps of all trajectories

    :param trajectories: as returned by run_policy()
    :return: None (mutates trajectories to add 'advantages' key)
    """
    for trajectory in trajectories:
        trajectory['advantages'] = trajectory['disc_sum_rew'] - trajectory['values']


def build_train_set(trajectories):
    """ Concatenates all trajectories into single NumPy array with first
     dimension = N = total time steps across all trajectories

    :param trajectories: trajectories after processing by add_disc_sum_rew(),
     add_value(), add_advantage()
    :return: 4-tuple of NumPy arrays
    observes: shape = (N, obs_dim)
    actions: shape = (N, act_dim)
    advantages: shape = (N,)
    disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])

    return observes, actions, advantages, disc_sum_rew


def disp_metrics(metrics):
    """Print metrics to stdout"""
    for key in metrics:
        print(key, ' ', metrics[key])


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
        trajectories = run_policy(env, policy, min_steps=1000)
        # calculate cum_sum_rew: all time steps
        add_disc_sum_rew(trajectories, gamma)
        # value prediction: all time steps
        add_value(trajectories, val_func)
        # calculate advantages: cum_sum_rew - v(s_t)
        add_advantage(trajectories)
        # policy update
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # print(observes.shape)
        # print(actions.shape)
        # print(advantages.shape)
        metrics = policy.update(observes, actions, advantages)
        print(metrics)
        # fit value function
        metrics.update(val_func.fit(observes, disc_sum_rew))
        disp_metrics(metrics)
        # view policy
        view_policy(env, policy)

if __name__ == "__main__":
    main()
