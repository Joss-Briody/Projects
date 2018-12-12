from __future__ import division
from __future__ import print_function
import sys
import os
import argparse
import math
import random
import pdb
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import time
import matplotlib.pyplot as plt

class batchQAgent(object):
	def __init__(self, alpha, gamma, action_size, state_size, has_hidden_layer):
		self.action_size = action_size
		self.state_size = state_size
		self.learning_rate = alpha
		self.discount_factor = gamma
		self.n_hidden = 100
		self.hidden = has_hidden_layer
		self.build_tf_graph()

	def build_tf_graph(self):
		"""Build TF computation graph"""
		if self.hidden == True:
			self.weights_hidden = tf.Variable(tf.random_uniform([self.state_size, self.n_hidden], 0, 0.01))
			self.weights_out = tf.Variable(tf.random_uniform([self.n_hidden, self.action_size], 0, 0.01))
			self.bias_hidden = tf.Variable(tf.random_uniform([self.n_hidden], 0, 0.01))
			self.bias_out = tf.Variable(tf.random_uniform([self.action_size], 0, 0.01))
		else:
			# self.weights = tf.Variable(tf.random_uniform([self.state_size, self.action_size], 0, 0.01))
			# self.bias = tf.Variable(tf.random_uniform([self.action_size], 0, 0.01))
			self.weights = tf.get_variable("weights0", [4, 2], initializer = tf.random_normal_initializer())
			self.bias = tf.get_variable("bias0", [2], initializer = tf.constant_initializer(0.01))

	def Q_net(self, state):
		"""returns the network's approximation of q(s,a)"""			
		if self.hidden:
			return tf.matmul(tf.nn.relu(tf.matmul(state, self.weights_hidden) + self.bias_hidden), self.weights_out) + self.bias_out 	
		else:
			return tf.matmul(state, self.weights) + self.bias

	def train_network(self, state, action, reward, next_state, alpha):
		"""performs batch q-learning to update q_net(s)"""
		self.q = self.Q_net(state)
		self.q_prime = self.Q_net(next_state)

		# use an indicator which ensures we do not bootstrap from terminal states, since reward is then -1
		is_state_terminal = (reward + 1)   
		maxQ_ = tf.reshape(tf.reduce_max(self.q_prime, reduction_indices=[1]), [-1, 1])
		q_a = tf.reshape(tf.gather_nd(self.q, action), [-1,1] )
		delta = tf.square(reward + self.discount_factor * tf.multiply(is_state_terminal, tf.stop_gradient(maxQ_)) - q_a) / 2
		self.loss = tf.reduce_mean(delta)
		self.opt = tf.train.RMSPropOptimizer(learning_rate=alpha).minimize(self.loss)
		return self.opt, self.loss


	def evaluate(self, env, n_evaluation_episodes, max_episode_length, sess, state):
		"""evaluate the agent's mean loss and return"""
		episode_lengths_list = []
		episode_returns_list = []

		q_ = self.Q_net(state)
		for episode in range(n_evaluation_episodes):
			obs = env.reset()
			for time_step in range(max_episode_length):
				q_k = sess.run(q_, feed_dict = {state: obs.reshape(-1,4)})
				act = np.argmax(q_k)
				obs, _, done, _ = env.step(act)
				if done: 
					episode_lengths_list.append(time_step+1)
					episode_returns_list.append(-1 * self.discount_factor ** time_step)
					break
		return np.mean(np.array(episode_lengths_list)), np.mean(np.array(episode_returns_list))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run-var', default='')
	parser.add_argument('--learning_rate')
	FLAGS, unparsed = parser.parse_known_args()

	if FLAGS.learning_rate == None:
			alpha = 0.001
			print("using default learning rate %d" % alpha)
	else:
		alpha = float(FLAGS.learning_rate)
		print("using learning rate %d" % alpha)

	MODEL_DIR = "../trained_models"
	
	PERF_DIR = "../performance"
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	if not os.path.exists(PERF_DIR):
		os.makedirs(PERF_DIR)

	gamma = 0.99
	n_epochs = 100
	test_n = 2
	n_episodes = 20
	batch_size = 64
	max_episode_length = 300
	rates_list = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]


	tf.reset_default_graph()
	print('alpha is: ', alpha)

	save_path = MODEL_DIR + '/' + 'batchQ_lin_lr_' + str(alpha)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	sampled_experience = 'sampled_experience_2' + str(max_episode_length) + '_episodes_.npy'

	#file structure: state1, state2, state3, state4, action, reward
	sampled_experience = np.load(sampled_experience) 

	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	state = tf.placeholder("float", [None, state_size])
	action = tf.placeholder(tf.int32, [None, action_size])
	next_state = tf.placeholder("float", [None, state_size])
	reward = tf.placeholder("float", [None, 1])

	agent = batchQAgent(alpha, gamma, action_size, state_size, has_hidden_layer=False)
	train_op, loss = agent.train_network(state, action, reward, next_state, alpha)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)

	mean_lengths = []
	mean_returns = []
	episode_loss = []

	with tf.Session() as sess:
		sess.run(init)

		############################# retraining and testing ##################################
		if FLAGS.run_var == 're-run':
			print('retraining and re-testing agent...')

			for e in range(n_epochs):
				print("epoch num: %d" % (e+1))
				ep_loss=0

				# test every test_n epochs for n_episodes
				if e % test_n == 0:
					sampled_length, sampled_return = agent.evaluate(env, n_episodes, max_episode_length, sess, state)
					mean_lengths.append(sampled_length)
					mean_returns.append(sampled_return)

				for batch_i in range(int(sampled_experience.shape[0] / batch_size) ):
					batch = sampled_experience[batch_i*batch_size:(batch_i+1)*batch_size, :]
					# use a as index to get qa = q_net[a]
					a = batch[:-1, state_size].reshape(-1, 1)
					a = np.append(np.arange(len(a)).reshape(-1, 1), a, axis=1) 
					o, l = sess.run([train_op, loss], feed_dict ={state: batch[:-1, :state_size],										
																  action: a,
																  next_state: batch[1:, :state_size],
																  reward: batch[:-1, -1].reshape(-1, 1)} )
					ep_loss += l
				episode_loss.append(ep_loss)

			saver.save(sess, save_path + '/batchq_' + str(alpha))

			#  save performance
			np.savetxt(PERF_DIR + '/loss_batch_' + str(alpha) + '.csv', episode_loss, delimiter=",")
			np.savetxt(PERF_DIR + '/lengths_batch_' + str(alpha) + '.csv', mean_lengths, delimiter=",")
			np.savetxt(PERF_DIR + '/returns_batch_' + str(alpha) + '.csv', mean_returns, delimiter=",")

	################# reload saved model and calculate mean lifetime #####################

		else: 
			print('re-evaluating on saved model')
			load_path  = save_path + '/batchq_' + str(alpha) + ".meta"
			num_repeats = 100
			saver = tf.train.import_meta_graph(load_path)
			saver.restore(sess, tf.train.latest_checkpoint(save_path)) 
			
			for t in range(num_repeats):
				sampled_length, sampled_return = agent.evaluate(env, n_episodes, max_episode_length, sess, state)
				mean_lengths.append(sampled_length)
				mean_returns.append(sampled_return)

			print('-----------------------------')
			print('mean survival length: %f' % np.mean(mean_lengths))
			print('variace in survival length: %f' % np.var(mean_lengths))
			print('mean return: %f' % np.mean(mean_returns))
			print('variace in return: %f' % np.var(mean_returns))
			print('-----------------------------')

		sess.close()

if __name__ == "__main__":
	main()