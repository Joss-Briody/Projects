from __future__ import division
from __future__ import print_function
import sys
import math
import random
import pdb
import numpy as np
import tensorflow as tf
import gym
import time

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
			self.weights = tf.Variable(tf.random_uniform([self.state_size, self.action_size], 0, 0.01))
			self.bias = tf.Variable(tf.random_uniform([self.action_size], 0, 0.01))


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
		self.loss = tf.reduce_sum(delta)
		self.opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.loss)

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

	gamma = 0.99
	n_epochs = 100
	test_n = 2
	n_episodes = 20
	batch_size = 64
	max_episode_length = 300
	rates_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

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

	for alpha in rates_list:
		print("training for learning rate %d" % alpha)

		agent = batchQAgent(alpha, gamma, action_size, state_size, has_hidden_layer=False)
		train_op, loss = agent.train_network(state, action, reward, next_state, alpha)

		init = tf.global_variables_initializer()
		mean_lengths = []
		mean_returns = []
		episode_loss = []

		with tf.Session() as sess:
			sess.run(init)

			for e in range(n_epochs):
				print("epoch num: %d" % (e+1))

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
																  reward: batch[1:, -1].reshape(-1, 1)} )
										
					episode_loss.append(l)

			sess.close()
			pdb.set_trace()
			print('hi')			

if __name__ == "__main__":
	random.seed(0)
	main()