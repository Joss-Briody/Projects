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
import random
import time

class DQNAgent(object):
	def __init__(self, state, action, next_state, reward, action_size, state_size, gamma):
		# self.state = state
		# self.action = action
		# self.next_state = next_state
		# self.reward = reward
		self.action_size = action_size
		self.state_size = state_size
		self.discount_factor = gamma 
		self.learning_rate = 0.0001
		self.epsilon = 0.05
		self.n_episodes = 2000
		self.max_episode_length = 100
		# self._Qval = None
		# self._Qval_next = None
		# self._optimize = None
		self._optimizeExists = False
		self._loss = None
		self._prediction_made = False
		self.build_graph()

	def build_graph(self):
		"""builds the tf computation graph used by the agent to approximate the q(s,a)"""
		self.n_hidden = 100
		self.weights_hidden = tf.get_variable("weights_hidden", [self.state_size, self.n_hidden], initializer = tf.random_normal_initializer())
		self.bias_hidden = tf.get_variable("bias_hidden", [self.n_hidden], initializer = tf.constant_initializer(0.1))

		self.weights_out = tf.get_variable("weights_out", [self.n_hidden, self.action_size], initializer = tf.random_normal_initializer())
		self.bias_out = tf.get_variable("bias_out", [self.action_size], initializer = tf.constant_initializer(0.1))

	def Q_net(self, state):
		""" q(s,a) ~= Q_net(s)[a] shape = [batch_size,2] """
		if not self._prediction_made: 
			Q  = tf.matmul(tf.nn.relu( tf.matmul(state, self.weights_hidden) + self.bias_hidden ), self.weights_out) + self.bias_out  
			self._Qval = Q	
			self._prediction_made = True
		return self._Qval

	def e_greedy(self, q):
		if np.random.rand() <= self.epsilon:
			return tf.random_uniform([self.action_size], maxval=1, dtype=tf.int64)
		else:
			return tf.argmax(q,1)

	def update_Qmodel(self, nextQ, reward, Q_a):
		if not self._optimizeExists:
			delta = (reward + discount*tf.stop_gradient(max_Q_next)-qa)



	def Q_learning(self, state, next_state, reward): # , env, sess):
		if not self._optimizeExists:
			q = self.Q_net(state)
			self.action = self.e_greedy(q)
			self.action_onehot = tf.one_hot(self.action, self.action_size, dtype=tf.float32)
			qa = tf.reshape(tf.reduce_sum(tf.multiply(q, self.action_onehot), axis=1), [-1,1])
			q_next = self.Q_net(next_state)
			max_q_next = tf.reshape(tf.reduce_max(q_next, axis=[1]),[-1,1])
			delta = (reward + self.discount_factor*tf.stop_gradient(max_q_next)-qa)

			optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
			self._loss = tf.reduce_mean(tf.square(delta)/2)
			self._optimize = optimizer.minimize(self._loss)
			self._optimizeExists = True

		return self._loss, self._optimize


		# for e in range(self.n_episodes):
		# 	obs = env.reset()
		# 	obs = np.reshape(obs, [-1, self.state_size])
		# 	for time in range(self.max_episode_length):
		# 		a = sess.run(self.action, feed_dict={state: obs})
		# 		next_obs, R, done, _ = env.step(a[0])
		# 		R = 0 if not done else -1
		# 		sess.run( self._optimize, feed_dict={state: obs, 
		# 											 next_state: next_obs.reshape(-1,4),
		# 											 reward: np.array(R).reshape(-1,1)} )
		# 		if done:
		# 			pdb.set_trace()
		# 			break

				# return sess.run(action, feed_dict={state: obs})
				# q_ = np.reshape(sess.run(q, feed_dict={state: obs} ),[2,-1] )
				
				# qa = q[action]
				# next_obs, R, done, _ = env.step(sess.run(action)
				# R = 0 if not done else -1
				# next_obs = next_obs.reshape(-1,4)
				# max_q_next = np.max(sess.run(self.Q_net(state), feed_dict={state: next_obs}), axis=1)
				# sess.run(self.update_Qmodel(nextQ, reward, Q_a), feed_dict={nextQ: max_q_next, reward: R, Q_a: qa})
				# if done:
				# 	break
				# obs = next_obs
		

def main():
	gamma = 0.99
	n_epochs = 100
	n_episodes = 2000
	max_eps_length = 300
	test_n = 5
	b_size = 100
	alpha_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	state = tf.placeholder(tf.float32, [None, state_size])
	next_state = tf.placeholder(tf.float32, [None, state_size])
	action = tf.placeholder(tf.int32, [None, action_size])
	# nextQ = tf.placeholder(tf.int32, [None, action_size])
	# Q_a = tf.placeholder(tf.int32, [None, 1])
	reward = tf.placeholder(tf.float32, [None, 1])
	
	agent = DQNAgent(state, action, next_state, reward, action_size, state_size, gamma)
	loss,_ = agent.Q_learning(state, next_state, reward)

	t0=time.time()
	init_op = tf.global_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init_op)
		obs = env.reset()
		loss_v = []
		for e in range(agent.n_episodes):
			loss_sum = 0
			obs = env.reset()
			obs = np.reshape(obs, [-1, agent.state_size])
			for time_step in range(agent.max_episode_length):
				a = sess.run(agent.action, feed_dict={state: obs})
				next_obs, R, done, _ = env.step(a[0])
				R = 0 if not done else -1
				t1,t2 = sess.run( [agent._optimize, agent._loss], feed_dict={state: obs, 
													 next_state: next_obs.reshape(-1,4),
													 reward: np.array(R).reshape(-1,1)} )

				loss_sum += t2

				if done:
					break

			loss_v.append(loss_sum)
		print(time.time()-t0)
		pdb.set_trace()


				


		test = agent.Q_learning(state, next_state, reward, env, sess) #feed_dict={state:obs.reshape(-1,4)})

		sess.close()
	pdb.set_trace()


if __name__ == "__main__":
	main()
# q = q_net(obs)
# action = e_greedy(q)
# qa = q[action]
# reward, discount, next_obs = env.step(action)
# max_Q_next = tf.reduce_max(q_net(next_obs))
# delta = (reward + discount*tf.stop_gradient(max_Q_next)-qa)
# loss = tf.square(delta)/2




