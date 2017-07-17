import gym
from gym import wrappers
from policy import *
from value_function import *
import scipy.signal
from utils import Logger, Scaler


def init_gym(env_name):
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


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        scaler:
        env: ai gym environment
        policy: policy with "sample" method
        animate: boolean, True uses env.render() method to animate episode

    Returns: 3-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    offset, scale = scaler.get_scale()
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float64).reshape((1, -1))
        unscaled_obs.append(obs)
        obs = (obs - offset) / (scale + 1e-4) / 3.0
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, min_steps, min_episodes):
    """ Run policy and collect data for a minimum of min_steps

    :param env: ai gym environment
    :param policy: policy with "sample" method
    :param min_steps: minimum number of samples to collect, completes current
    episode after min_steps reached
    :return: list dictionaries, 1 dictionary per episode. Dict key/values:
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (undiscounted) rewards from episode

    Args:
        min_episodes:
        obs_scaler:
    """
    steps, episodes = (0, 0)
    trajectories = []
    while not (steps >= min_steps and episodes >= min_episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        steps += observes.shape[0]
        episodes += 1
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update_scale(unscaled)
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': steps})

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

    Args:
        gamma:
    """
    for trajectory in trajectories:
        rewards = trajectory['rewards'] * (1 - gamma)
        disc_sum_rew = scipy.signal.lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func, gamma):
    """ Adds estimated value to all timesteps of all trajectories

    :param trajectories: as returned by run_policy()
    :return: None (mutates trajectories to add 'values' key)

    Args:
        val_func:
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
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger):
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.max(observes),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.max(actions),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.max(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.max(disc_sum_rew),
                })


def main(num_iter=5000,
         gamma=0.995):

    env_name = 'Hopper-v1'
    env, obs_dim, act_dim = init_gym(env_name)
    scaler = Scaler(obs_dim)
    logger = Logger(logname=env_name)
    env = wrappers.Monitor(env, '/tmp/hopper-experiment-1', force=True)
    lin_val_func = ValueFunction(obs_dim)
    val_func = LinearValueFunction()
    policy = Policy(obs_dim, act_dim)
    run_policy(env, policy, scaler, logger, min_steps=500, min_episodes=5)
    for i in range(num_iter):
        logger.log({'_Iteration': i})
        trajectories = run_policy(env, policy, scaler, logger, min_steps=5000, min_episodes=20)
        add_value(trajectories, val_func, gamma)
        add_disc_sum_rew(trajectories, gamma)
        add_advantage(trajectories)
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger)
        policy.update(observes, actions, advantages, logger)
        val_func.fit(observes, disc_sum_rew, logger)
        lin_val_func.fit(observes, disc_sum_rew, logger)
        logger.write(display=True)
    logger.close()


if __name__ == "__main__":
    main()
