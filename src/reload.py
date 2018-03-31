import tensorflow as tf
import _pickle as pickle
import gym, sys, os
from gym import wrappers
import numpy as np


def getScaleAndOffset(env_name):
	file_name = './saved_models/' + env_name + '/scale_and_offset.pkl'
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data['SCALE'], data['OFFSET']


def reload_and_run(env_name):
	directory_to_load_from = './saved_models/' + env_name + '/'
	if not os.path.exists(directory_to_load_from):
		print('Trained model for ' + env_name + ' doesn\'t exist. Program is exiting now...')
		exit(0)
	imported_meta = tf.train.import_meta_graph(directory_to_load_from + 'final.meta')
	sess = tf.Session()
	imported_meta.restore(sess, tf.train.latest_checkpoint(directory_to_load_from))
	graph = tf.get_default_graph()
	scaled_observation_node = graph.get_tensor_by_name('obs:0')
	output_action_node = graph.get_tensor_by_name('output_action:0')
	scale, offset = getScaleAndOffset(env_name)
	env = gym.make(env_name)
	#env = wrappers.Monitor(env, aigym_path, force=True)
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
	env_name = sys.argv[1]
	reload_and_run(env_name)