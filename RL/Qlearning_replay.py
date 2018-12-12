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


class experience_replay_buffer(object):
	def __init__(self, replay_size = 1000):
		self.replay_buffer = []
		self.buffer_size = replay_size

	def remember_experience(self, experience):
		"""adds a transition experience to the buffer"""
		if len(self.replay_buffer) >= self.buffer_size:
			self.replay_buffer[0:(len(experience)+len(self.replay_buffer)) - self.buffer_size] = []
		self.replay_buffer.append(experience)

	def sample(self, size):
		"""randomly sample a fixed size from past experiences"""
		return np.reshape(np.array(random.sample(self.replay_buffer, size)),[size,4]) 


class DQNagent():
	def __init__(self, learning_rate, state_size, action_size):
		""" create and store agent's variables """
		self.state =  tf.placeholder(dtype=tf.float32, shape=[None, state_size])
		self.next_state =  tf.placeholder(dtype=tf.float32, shape=[None, state_size])
		self.actions = tf.placeholder(dtype=tf.int32, shape=[None, action_size])
		self.reward = tf.placeholder("float", [None, 1])
		self.learning_rate = learning_rate
		self.n_units = 100
		self.discount_factor = 0.99
		self.state_size = state_size
		self.action_size = action_size
		self.max_len_episode = 300
		self.n_evaluation_episodes = 20
		self.experience_replay = experience_replay_buffer()
		self.Build_Qnet_graph()
		self.Build_Qlearning_graph()

	def Build_Qnet_graph(self):
		"""Build tf graph for the q-value function approximtor"""
		self.weights = tf.get_variable("weights", [self.state_size, self.n_units], initializer = tf.random_normal_initializer())
		self.bias = tf.get_variable("bias", [self.n_units], initializer = tf.constant_initializer(0.1))
		self.weights_out = tf.get_variable("weights_out", [self.n_units, self.action_size], initializer = tf.random_normal_initializer())
		self.bias_out = tf.get_variable("bias_out", [self.action_size], initializer = tf.constant_initializer(0.1))
		self.Qout = tf.matmul(tf.nn.relu(tf.matmul(self.state, self.weights) + self.bias), self.weights_out) + self.bias_out
		self.Qout_next = tf.matmul(tf.nn.relu(tf.matmul(self.next_state, self.weights) + self.bias), self.weights_out) + self.bias_out

	def Build_Qlearning_graph(self):
		"""performs batch q-learning to update q_net(s)"""
		self.maxQ = tf.argmax(self.Qout,1)
		# use an indicator to ensure we don't bootstrap from terminal states (where reward is -1)
		is_state_terminal = (self.reward + 1)   
		maxQ_ = tf.reshape(tf.reduce_max(self.Qout_next, reduction_indices=[1]), [-1, 1])
		q_a = tf.reshape(tf.gather_nd(self.Qout, self.actions), [-1,1] )
		delta = tf.square(self.reward + self.discount_factor * tf.multiply(is_state_terminal, tf.stop_gradient(maxQ_)) - q_a) / 2
		self.loss = tf.reduce_mean(delta)
		self.updateModel = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def epsilon_greedy(self, a, eps, env):
		"""act with an e-greey policy derived from q[a]"""
		if np.random.rand() < eps:
			return env.action_space.sample()
		else:
			return a[0]

	def evaluate(self, env, sess):
		"""evaluate the agent's mean loss and return using a greedy policy"""
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

def training_episode_with_replay(agent, epsilon, env, sess, state):
	"""train using a replay buffer for one episode of experience"""
	loss = 0
	state = np.reshape(state, (-1,4))
	for t in range(agent.max_len_episode):
		a, Qall = sess.run([agent.maxQ, agent.Qout],feed_dict={agent.state: state})
		a = agent.epsilon_greedy(a, epsilon, env)
		next_state, R, done, _  = env.step(a)
		R = 0 if not done else -1
		next_state = np.reshape(next_state, (-1,4))
		transition_exp = (state, a, R, next_state)
		agent.experience_replay.remember_experience(transition_exp)

		if len(agent.experience_replay.replay_buffer) >= agent.experience_replay.buffer_size:
			sampled_experience = agent.experience_replay.sample(128)
			states_batch = np.stack(sampled_experience[:,0], axis=0).reshape(-1,4)
			next_states_batch = np.stack(sampled_experience[:,-1], axis=0).reshape(-1,4)
			rewards_batch = np.stack(sampled_experience[:,2], axis=0)
			# q_a = q[a] so use actions tensor to index q(state)
			actions_batch = sampled_experience[:,1].reshape(-1,1)
			actions_batch = np.append(np.arange(len(actions_batch)).reshape(-1, 1), actions_batch, axis=1)
		
			l, _ = sess.run([agent.loss,agent.updateModel], feed_dict={agent.state: states_batch,
		                    									   	   agent.actions: actions_batch,
		                    									   	   agent.next_state: next_states_batch,
		                    									   	   agent.reward: rewards_batch.reshape(-1,1)})
			loss += l
		state = next_state
		if done:
			break
	return t, loss

def run_Q_learning():
	"""train and test a q-learning agent with replay"""
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

	# learning parameters
	n_episodes = 2000
	n_test_episode = 10
	print_frequency = 500
	test_freq = 20
	learning_rate = alpha
	n_runs = 10

	tf.reset_default_graph()
	agent = DQNagent(learning_rate, state_size, action_size)
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	mean_lengths = []
	mean_returns = []

	with tf.Session() as sess:
		sess.run(init)
		start_time = time.time()

		startE = 0.5
		endE = 0.005
		n_to_decay = 2000
		delta_epsilon = (startE - endE)/n_to_decay
		epsilon = startE
		episodes_loss, episodes_len, episodes_return = [], [], []
		totals_steps = 0

		# Derive an epsilon greedy policy from the action-value function
		for e in range(n_episodes):	
			state = env.reset()
			t, loss = training_episode_with_replay(agent, epsilon, env, sess, state)
			episodes_loss.append(loss/t)
			if epsilon > endE:
				epsilon = epsilon - delta_epsilon
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