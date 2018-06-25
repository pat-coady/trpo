"""
Reload and reuse the trained PPO/TRPO policies

Written by Sanjay Thakur (sanjaykthakur.com)
"""

import tensorflow as tf
import _pickle as pickle
import gym, os, argparse
from gym import wrappers
import numpy as np
from datetime import datetime


def get_scale_and_offset(env_name: str):
    """
    Returns the values that were used to normalize to derive the saved PPO/TRPO models.

    Parameters:
        env_name (string):  Name of the environment whose normalization values are to be fetched

    Returns: The scale and the offset values
    """
    file_name = './saved_models/' + env_name + '/scale_and_offset.pkl'
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data['SCALE'], data['OFFSET']


def reload_and_run(env_name: str):
    """
    Loads the trained models, values for normalization, and runs them on fresh environment instances

    Parameters:
        env_name (string):  Name of the environment to load and run
    """
    directory_to_load_from = './saved_models/' + env_name + '/'
    if not os.path.exists(directory_to_load_from):
        print('Trained model for ' + env_name + ' doesn\'t exist. Run train.py first. Program is exiting now...')
        exit(0)
    imported_meta = tf.train.import_meta_graph(directory_to_load_from + 'final.meta')
    sess = tf.Session()
    imported_meta.restore(sess, tf.train.latest_checkpoint(directory_to_load_from))
    graph = tf.get_default_graph()
    scaled_observation_node = graph.get_tensor_by_name('obs:0')
    output_action_node = graph.get_tensor_by_name('output_action:0')
    scale, offset = get_scale_and_offset(env_name)
    env = gym.make(env_name)
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    aigym_path = os.path.join('/tmp', env_name, now)
    env = wrappers.Monitor(env, aigym_path, force=True)
    observation = env.reset()
    done = False
    total_reward = 0.
    time_step = 0.
    while not done:
        env.render()
        observation = observation.astype(np.float32).reshape((1, -1))
        observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
        action = sess.run(output_action_node, feed_dict={scaled_observation_node: (observation - offset) * scale})
        observation, reward, done, info = env.step(action)
        total_reward += reward
        time_step += 1e-3
    print('Episodic reward at this episode is ' + str(total_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reload and run trained PPO based policies on OpenAI Gym environment')
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    args = parser.parse_args()
    reload_and_run(args.env_name)
