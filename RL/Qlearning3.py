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
import argparse


class DQNagent():
	def __init__(self, learning_rate, n_units, state_size, action_size):
		""" creat and store agent's variables """
		self.state =  tf.placeholder(dtype=tf.float32, shape=[None,4])
		self.targetQ = tf.placeholder(dtype=tf.float32, shape=[None])
		self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
		self.learning_rate = learning_rate
		self.n_units = n_units
		self.discount_factor = 0.99
		self.state_size = state_size
		self.action_size = action_size
		self.max_len_episode = 300
		self.n_evaluation_episodes = 20
		self.Build_Qnet_graph()
		self.Build_Qlearning_graph()

	def Build_Qnet_graph(self):
		"""Build tf graph for the q-value function approximtor"""
		self.weights = tf.get_variable("weights", [self.state_size, self.n_units], initializer = tf.random_normal_initializer())
		self.bias = tf.get_variable("bias", [self.n_units], initializer = tf.constant_initializer(0.1))
		self.weights_out = tf.get_variable("weights_out", [self.n_units, self.action_size], initializer = tf.random_normal_initializer())
		self.bias_out = tf.get_variable("bias_out", [self.action_size], initializer = tf.constant_initializer(0.1))
		self.Qout = tf.matmul(tf.nn.relu(tf.matmul(self.state, self.weights) + self.bias), self.weights_out) + self.bias_out

	def Build_Qlearning_graph(self):
		"""Build tf graph to implement online q-learning"""
		self.maxQ = tf.argmax(self.Qout,1)
		self.actions_onehot = tf.one_hot(self.actions,self.action_size,dtype=tf.float32)
		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),axis=1)
		td_error = tf.square(tf.stop_gradient(self.targetQ) - self.Q) / 2
		l = tf.reduce_mean(td_error)
		self.loss = l
		self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.updateModel = self.trainer.minimize(self.loss)

	def epsilon_greedy(self, a, eps, env):
		"""act with an e-greey policy derived from q[a]"""
		if np.random.rand() < eps:
			return env.action_space.sample()
		else:
			return a[0]

	def evaluate(self, env, sess):
		"""evaluate the agent's mean loss and return"""
		episode_lengths_list = []
		episode_returns_list = []

		for episode in range(self.n_evaluation_episodes):
			obs = env.reset()
			for time_step in range(self.max_len_episode):
				q_k = sess.run(self.Qout, feed_dict = {self.state: obs.reshape(-1,4)})
				act = np.argmax(q_k)
				obs, _, done, _ = env.step(act)
				if done: 
					episode_lengths_list.append(time_step+1)
					episode_returns_list.append(-1 * self.discount_factor ** time_step)
					break
		return np.mean(np.array(episode_lengths_list)), np.mean(np.array(episode_returns_list))

def training_episode(agent, epsilon, env, sess, state):
	loss = 0
	for t in range(agent.max_len_episode):
		a, Qall = sess.run([agent.maxQ, agent.Qout],feed_dict={agent.state: state.reshape(-1,4)})
		a = agent.epsilon_greedy(a, epsilon, env)
		next_state, R, done, _  = env.step(a)
		R = 0 if not done else -1

		Qout = sess.run(agent.Qout,feed_dict={agent.state: next_state.reshape(-1,4)})
		maxQ = np.amax(Qout, axis=1)
		# do not bootstrap on terminal states
		targetQ = R + (1+R) * agent.discount_factor * maxQ
		l, _ = sess.run([agent.loss,agent.updateModel], feed_dict={agent.state: state.reshape(-1,4),
		                  										   agent.targetQ: np.array([targetQ]).reshape(-1),
		                    									   agent.actions: np.array(a).reshape(-1)})
		loss += l
		state = next_state
		if done:
			break

	return t, loss


def run_Q_learning():

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run-var', default='')
	parser.add_argument('--learning_rate')
	FLAGS, unparsed = parser.parse_known_args()

	if FLAGS.learning_rate == None:
		alpha = 0.0001
		print("using default learning rate %d" % alpha)
	else:
		alpha = float(FLAGS.learning_rate)
		print("using learning rate %d" % alpha)
	
	env = gym.make('CartPole-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	n_episodes = 2000
	n_test_episode = 10
	print_frequency = 500
	test_freq = 20
	lr_list = [0.00001, 0.0001, 0.0005, 0.01]
	learning_rate = alpha
	n_units = 100
	nunits_list = [100,]
	n_runs = 10

	tf.reset_default_graph()
	agent = DQNagent(alpha, n_units, state_size, action_size)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	mean_lengths = []
	mean_returns = []

	with tf.Session() as sess:
		sess.run(init)
		start_time = time.time()
		episodes_loss, episodes_len, episodes_return = [], [], []
		totals_steps = 0

		startE = 0.5
		endE = 0.005
		n_to_decay = 2000
		delta_epsilon = (startE - endE)/n_to_decay
		epsilon = startE

		# Derive an epsilon greedy policy from the action-value function
		for e in range(n_episodes):
			
			state = env.reset()
			t, loss = training_episode(agent, epsilon, env, sess, state)

			if epsilon > endE:
				epsilon = epsilon - delta_epsilon
			episodes_loss.append(loss/t)

			if e % print_frequency == 0:
				print("Time taken for episodes {} to {} was {:2f}s".format(max(0,e+1-print_frequency),e,time.time()-start_time))
				print("Train loss: ", (loss/(t+1)))
				print("\n")

			if e % test_freq == 0:
				test_length, test_return = agent.evaluate(env, sess)
				episodes_len.append(test_length)
				episodes_return.append(test_return)

				if e % print_frequency == 0:
					print("Running greedy policy after {} episodes:".format(e))
					print("Mean episode length:  {:.3f}\t Mean episode return:  {:.3f}".format(test_length, test_return))
					print("\n")

	sess.close()

	pdb.set_trace()	

def main():
	run_Q_learning()	


if __name__ == "__main__":
	main()


