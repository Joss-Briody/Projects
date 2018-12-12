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
	def __init__(self, learning_rate, state_size, action_size, network_scope):
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
		self.Build_Qnet_graph(network_scope)


	def Build_Qnet_graph(self, network_scope):
		"""Build tf graph for the q-value function approximtor"""
		with tf.variable_scope(network_scope):
			self.weights = tf.get_variable("weights", [self.state_size, self.n_units], initializer = tf.random_normal_initializer(stddev=0.01))
			self.bias = tf.get_variable("bias", [self.n_units], initializer = tf.constant_initializer(0.1))
			self.weights_out = tf.get_variable("weights_out", [self.n_units, self.action_size], initializer = tf.random_normal_initializer(stddev=0.01))
			self.bias_out = tf.get_variable("bias_out", [self.action_size], initializer = tf.constant_initializer(0.1))
			self.Qout = tf.matmul(tf.nn.relu(tf.matmul(self.state, self.weights) + self.bias), self.weights_out) + self.bias_out
			self.Qout_next = tf.matmul(tf.nn.relu(tf.matmul(self.next_state, self.weights) + self.bias), self.weights_out) + self.bias_out
			self.maxQ = tf.argmax(self.Qout,1)

	def Build_Q_learning_graph(self, target_agent):
		"""performs batch q-learning to update q_net(s)"""
		amaxQ = tf.argmax(self.Qout_next,1)
		amaxQ_onehot = tf.one_hot(amaxQ, self.action_size, dtype=tf.float32)
		maxQ_next = tf.reshape(tf.reduce_sum(tf.multiply(target_agent.Qout_next, amaxQ_onehot), axis=1), [-1,1])
		is_state_terminal = (self.reward + 1)   
		q_a = tf.reshape(tf.gather_nd(self.Qout, self.actions), [-1,1] )
		delta = tf.square(self.reward + self.discount_factor * tf.multiply(is_state_terminal, tf.stop_gradient(maxQ_next)) - q_a) / 2
		self.loss = tf.reduce_mean(delta)
		self.updateModel = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

	def epsilon_greedy(self, Qall_A, Qall_B, eps, env):
		"""act with an e-greey policy derived from q[a]"""
		if np.random.rand() < eps:
			return env.action_space.sample()
		else:
			return np.argmax(Qall_A+Qall_B)

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


def training_episode_with_replay_doubleQ(agentA, agentB, epsilon, env, sess, state):
	"""train using a replay buffer and target network for one episode of experience"""
	loss1 = 0
	loss2 = 0
	state = np.reshape(state, (-1,4))
	for t in range(agentA.max_len_episode):
		Qall_A, Qall_B = sess.run([agentA.Qout, agentB.Qout], feed_dict={agentA.state: state, agentB.state: state})
		a = agentA.epsilon_greedy(Qall_A, Qall_B, epsilon, env)
		next_state, R, done, _  = env.step(a)
		R = 0 if not done else -1
		next_state = np.reshape(next_state, (-1,4))
		transition_exp = (state, a, R, next_state)
		agentA.experience_replay.remember_experience(transition_exp)

		if len(agentA.experience_replay.replay_buffer) >= agentA.experience_replay.buffer_size:
			sampled_experience = agentA.experience_replay.sample(128)
			states_batch = np.stack(sampled_experience[:,0], axis=0).reshape(-1,4)
			next_states_batch = np.stack(sampled_experience[:,-1], axis=0).reshape(-1,4)
			rewards_batch = np.stack(sampled_experience[:,2], axis=0)
			# q_a = q[a] so use actions tensor to index q(state)
			actions_batch = sampled_experience[:,1].reshape(-1,1)
			actions_batch = np.append(np.arange(len(actions_batch)).reshape(-1, 1), actions_batch, axis=1)
		
			if np.random.rand() < 0.5:
				l, _ = sess.run([agentA.loss, agentA.updateModel], 
						feed_dict = {	agentA.state: states_batch,
										agentB.state: states_batch,
									    agentA.next_state: next_states_batch,
									    agentB.next_state: next_states_batch,
								   	    agentA.actions: actions_batch,
								   	    agentB.actions: actions_batch, 
								   	   	agentA.reward: rewards_batch.reshape(-1,1),
								   	    agentB.reward: rewards_batch.reshape(-1,1)	})
				loss1 += l
			else:
				l, _ = sess.run([agentB.loss, agentB.updateModel], 
						feed_dict = {	agentA.state: states_batch,
										agentB.state: states_batch,
									    agentA.next_state: next_states_batch,
									    agentB.next_state: next_states_batch,
								   	    agentA.actions: actions_batch,
								   	    agentB.actions: actions_batch, 
								   	   	agentA.reward: rewards_batch.reshape(-1,1),
								   	    agentB.reward: rewards_batch.reshape(-1,1)	})
				loss2 += l


		state = next_state
		if done:
			break
	return t, loss1, loss2

def run_doubleQ():
	"""train and test a q-learning agent with replay"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run-var', default='')
	parser.add_argument('--learning_rate')
	FLAGS, unparsed = parser.parse_known_args()

	if FLAGS.learning_rate == None:
		alpha = 0.0005
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
	agentA = DQNagent(learning_rate, state_size, action_size, 'A')
	agentB = DQNagent(learning_rate, state_size, action_size, 'B')
	tf_vars = tf.trainable_variables()
	agentA.Build_Q_learning_graph(agentB)
	agentB.Build_Q_learning_graph(agentA)
	update_operation = [tf_vars[ix+len(tf_vars)//2].assign(var.value()) for ix, var in enumerate(tf_vars[0:len(tf_vars)//2])]
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	mean_lengths = []
	mean_returns = []

	with tf.Session() as sess:
		sess.run(init)
		# map(lambda x: sess.run(x), update_operation)
		start_time = time.time()

		startE = 0.5
		endE = 0.005
		n_to_decay = 2000
		delta_epsilon = (startE - endE)/n_to_decay
		epsilon = startE
		episodes_loss1, episodes_loss2, episodes_len, episodes_return = [], [], [], []
		totals_steps = 0

		# Derive an epsilon greedy policy from the action-value function
		for e in range(n_episodes):	
			state = env.reset()

			t, loss1, loss2 = training_episode_with_replay_doubleQ(agentA, agentB, epsilon, env, sess, state)
			episodes_loss1.append(loss1/t)
			episodes_loss2.append(loss2/t)
			if epsilon > endE:
				epsilon = epsilon - delta_epsilon
			if e % print_frequency == 0:
				print("Time taken for episodes {} to {} was {:2f}s".format(max(0,e+1-print_frequency),e,time.time()-start_time))
				print("Train loss1: ", (loss1/(t+1)))
				print("Train loss1: ", (loss2/(t+1)))
				print("\n")
			if e % test_freq == 0:
				test_length, test_return = agentA.evaluate(env, sess)
				episodes_len.append(test_length)
				episodes_return.append(test_return)
			if e % print_frequency == 0:
				print("Running greedy policy after {} episodes:".format(e))
				print("Mean episode length:  {:.3f}\t Mean episode return:  {:.3f}".format(test_length, test_return))
				print("\n")

	sess.close()

	pdb.set_trace()	

def main():
	run_doubleQ()	

if __name__ == "__main__":
	main()