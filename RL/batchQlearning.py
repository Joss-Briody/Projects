from __future__ import division
from __future__ import print_function
from collections import deque
import gym
import sys
from gym import wrappers
import numpy as np
import random
import tensorflow as tf
import pdb
import time

class batchQAgent(object):
	def __init__(self, state, action, next_state, reward, action_size, state_size, gamma, isLinear):
		self.state = state
		self.action = action
		self.next_state = next_state
		self.reward = reward
		self.action_size = action_size
		self.state_size = state_size
		self.linear_representation = isLinear
		self.discount_factor = gamma 
		self.learning_rate = 0.0001
		self._Qval = None
		self._Qval_next = None
		self._optimize = None
		self._optimizeExists = False
		self._loss = None
		self._prediction_made = False
		self.build_graph()

	def build_graph(self):
		"""builds the tf computation graph used by the agent to approximate the q(s,a)"""
		if self.linear_representation: 		
			self.weights = tf.get_variable("weights", [self.state_size, self.action_size], initializer = tf.random_normal_initializer())
			self.bias = tf.get_variable("bias", [self.action_size], initializer = tf.constant_initializer(0.1))

		else:
			self.n_hidden = 100
			self.weights_hidden = tf.get_variable("weights_hidden", [self.state_size, self.n_hidden], initializer = tf.random_normal_initializer())
			self.bias_hidden = tf.get_variable("bias_hidden", [self.n_hidden], initializer = tf.constant_initializer(0.1))

			self.weights_out = tf.get_variable("weights_out", [self.n_hidden, self.action_size], initializer = tf.random_normal_initializer())
			self.bias_out = tf.get_variable("bias_out", [self.action_size], initializer = tf.constant_initializer(0.1))

	def Q_net(self, state):
		""" q(s,a) ~= Q_net(s)[a] shape = [batch_size,2] """
		if not self._prediction_made: 
			if self.linear_representation: 		
				Q = tf.matmul(state, self.weights) + self.bias
			else:
				Q  = tf.matmul(tf.nn.relu( tf.matmul(state, self.weights_hidden) + self.bias_hidden ), self.weights_out) + self.bias_out  
			self._Qval = Q	
			self._prediction_made = True

		return self._Qval

	@property
	def train_Q_net(self):
		"""learn the action-value function via batch Q-learning"""
		if not self._optimizeExists:
			Q = self.Q_net(self.state)
			Q_ = self.Q_net(self.next_state)

			maxQ_ = tf.reshape(tf.reduce_max(Q_, axis = [1]), [-1,1])
			self.action_onehot = tf.one_hot(self.action, self.action_size, dtype=tf.float32)
			# Q_t = tf.reshape(tf.reduce_sum(tf.multiply(Q, self.action_onehot), axis=1), [-1,1])
			Q_t = tf.reshape(tf.gather_nd(Q, self.action), [-1,1] )
			is_state_terminal = (self.reward + 1) # this is an indicator which ensures we do not bootstrap from terminal states, since reward is then -1
			delta = tf.square(self.reward + self.discount_factor * tf.multiply(is_state_terminal, tf.stop_gradient(maxQ_)) - Q_t) / 2
			optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
			self._loss = tf.reduce_mean(delta)
			self._optimize = optimizer.minimize(self._loss)
			self._optimizeExists = True

		return self._optimize, self._loss

	def evaluate(self, env, n_evaluation_episodes, max_episode_length, sess, state):
		"""evaluate the agent's mean loss and return"""
		episode_lengths_list = []
		episode_returns_list = []

		# q_ = self.Q_net(state)
		for episode in range(n_evaluation_episodes):
			obs = env.reset()
			for time_step in range(max_episode_length):
				q_k = sess.run(self.Q_net(state), feed_dict = {state: obs.reshape(-1,4)})
				act = np.argmax(q_k)
				obs, _, done, _ = env.step(act)
				if done: 
					episode_lengths_list.append(time_step+1)
					episode_returns_list.append(-1 * self.discount_factor ** time_step)
					break

		return np.mean(np.array(episode_lengths_list)), np.mean(np.array(episode_returns_list))


def main():
	# Training parameters
	gamma = 0.99
	n_epochs = 200
	n_episodes = 500
	max_eps_length = 300
	test_n = 10
	b_size = 100
	alpha_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

	sampled_experience ='sampled_experience_300_episodes_.npy'
	sampled_experience = np.asarray(np.load(sampled_experience))

	action_history = np.concatenate(sampled_experience[0], axis=0)
	action_index   = np.asarray([np.arange(action_history.shape[0]), action_history])
	state_history  = np.concatenate(sampled_experience[1], axis=0)
	reward_history = np.concatenate(sampled_experience[2], axis=0)
	n_interactions = len(state_history)

	alpha = alpha_list[0]
	print("#------------- Running for learning rate %f" % alpha)

	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	# placeholders for transition quadruple
	state = tf.placeholder(tf.float32, [None, state_size])
	next_state = tf.placeholder(tf.float32, [None, state_size])
	action = tf.placeholder(tf.int32, [None, action_size])
	reward = tf.placeholder(tf.float32, [None, 1])
	
	agent = batchQAgent(state, action, next_state, reward, action_size, state_size, gamma, False)
	q = agent.Q_net(state)
	train_op, loss = agent.train_Q_net
	
	mean_lengths = []
	mean_returns = []
	episode_loss = []

	t0=time.time()
	init_op = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init_op)

		for epoch in range(n_epochs):
			print("Epoch number: %d" % (epoch+1))
		
			# 100 test episodes every test_n epochs
			if epoch % test_n == 0:
				sampled_length, sampled_return = agent.evaluate(env, n_episodes, max_eps_length, sess, state)
				mean_lengths.append(sampled_length)
				mean_returns.append(sampled_return)


			for i in range(int(n_interactions / b_size)):
				actions = action_history[i*b_size:(i+1)*b_size -1]
				actions = np.append(np.arange(len(actions)).reshape(-1, 1), actions.reshape(-1, 1), axis=1) 
				states = state_history[i*b_size:(i+1)*b_size -1, :]
				next_states = state_history[i*b_size+1:(i+1)*b_size, :]
				rewards = reward_history[i*b_size:(i+1)*b_size -1]
		
				o, l = sess.run([train_op, loss], feed_dict={ state: states.reshape(-1,4), 
															  action: actions, 
															  next_state: next_states.reshape(-1,4), 
															  reward: rewards.reshape(-1,1) })

			episode_loss.append(l)

		sess.close()
	print(time.time()-t0)
	pdb.set_trace()

if __name__ == "__main__":
	# random.seed(0)
	main()