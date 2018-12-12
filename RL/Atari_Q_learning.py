from __future__ import division
from __future__ import print_function
from collections import deque
import gym
import sys
import os
from gym import wrappers
import numpy as np
import random
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
import pdb
import random
import time
import argparse

def clip_reward(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else: 
		return 0

def preprocess_state(frame):
	resized = resize(frame, (28,28), preserve_range=True)
	return rgb2gray(resized)#.astype(np.uint8)

def get_init_state(env):
	state = np.empty([28,28,4])
	obs = env.reset()
	for i in range(4):
		state[:,:,i] = preprocess_state(obs)
		action = env.action_space.sample()
		obs, _, _, _ = env.step(action)
	return obs, state

def new_state(state, observation):
	"""update last 4 frames"""
	state = np.append(state, preprocess_state(observation).reshape(28, 28, 1), axis=2)
	new_state = state[:, :, 1:]
	return new_state

class experience_replay_buffer(object):
	def __init__(self, replay_size = 100000):
		self.replay_buffer = []
		self.buffer_size = replay_size

	def remember_experience(self, experience):
		"""adds a transition experience to the buffer"""
		if len(self.replay_buffer) >= self.buffer_size:
			self.replay_buffer[0:(len(experience)+len(self.replay_buffer)) - self.buffer_size] = []
		self.replay_buffer.append(experience)

	def sample(self, size):
		"""randomly sample a fixed size from past experiences"""
		return np.reshape(np.array(random.sample(self.replay_buffer, size)),[size,5]) 


class DQNagent():
	def __init__(self, learning_rate, state_size, action_size, network_scope):
		"""create and store agent's variables"""
		self.state =  tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 4])
		self.next_state =  tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 4])
		self.actions = tf.placeholder(dtype=tf.int32, shape=[None,1]) 
		self.reward = tf.placeholder("float", [None, 1])
		self.is_state_terminal = tf.placeholder("float", [None, 1])
		self.learning_rate = learning_rate
		self.discount_factor = 0.99
		self.state_size = state_size
		self.action_size = action_size
		self.n_evaluation_episodes = 100
		self.update_target_frequency = 5000
		self.test_freq = 50000
		self.experience_replay = experience_replay_buffer()
		self.Build_Qnet_graph(network_scope)

	def Build_Qnet_graph(self, network_scope):
		"""Build tf graph for the q-value function approximtor"""
		with tf.variable_scope(network_scope):
			self.W_conv1 = tf.get_variable("conv_w1", [6, 6, 4, 16], initializer = tf.truncated_normal_initializer())
			self.b_conv1 = tf.get_variable("conv_b1", [16], initializer = tf.constant_initializer(0.1))
			self.W_conv2 = tf.get_variable("conv_w2", [4, 4, 16, 32], initializer = tf.truncated_normal_initializer())
			self.b_conv2 = tf.get_variable("conv_b2", [32], initializer = tf.constant_initializer(0.1))
			self.W_fc = tf.get_variable("fc_w", [7*7*32, 256], initializer = tf.truncated_normal_initializer())
			self.b_fc = tf.get_variable("fc_b", [256], initializer = tf.truncated_normal_initializer())
			self.W_linear = tf.get_variable("lin_w", [256, self.action_size], initializer = tf.truncated_normal_initializer())
			self.b_linear = tf.get_variable("lin_b", [self.action_size], initializer = tf.truncated_normal_initializer())

			self.conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.state, self.W_conv1, strides=[1, 2, 2, 1], padding='SAME'), self.b_conv1))
			self.conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME'), self.b_conv2))
			self.conv_flat = tf.reshape(self.conv2,  [-1, 7*7*32])
			self.fully_connected = tf.nn.relu(tf.matmul(self.conv_flat, self.W_fc) + self.b_fc)
			self.Qout = tf.matmul(self.fully_connected, self.W_linear) + self.b_linear

			self.conv1_next = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.next_state, self.W_conv1, strides=[1, 2, 2, 1], padding='SAME'), self.b_conv1))
			self.conv2_next = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.conv1_next, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME'), self.b_conv2))
			self.conv_flat_next = tf.reshape(self.conv2_next,  [-1, 7*7*32])
			self.fully_connected_next = tf.nn.relu(tf.matmul(self.conv_flat_next, self.W_fc) + self.b_fc)
			self.Qout_next = tf.matmul(self.fully_connected_next, self.W_linear) + self.b_linear

	def Build_Qlearning_graph(self, target_agent):
		"""performs batch q-learning to update q_net(s)"""
		self.maxQ = tf.argmax(self.Qout,1)
		maxQ_next = tf.reshape(tf.reduce_max(target_agent.Qout_next, reduction_indices=[1]), [-1, 1])
		self.actions_onehot = tf.one_hot(self.actions,self.action_size,dtype=tf.float32)
		q_a = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),axis=1)
		delta = tf.square(self.reward + self.discount_factor * tf.multiply(self.is_state_terminal, tf.stop_gradient(maxQ_next)) - q_a) / 2
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
			obs, state = get_init_state(env)
			done = False
			t=0
			scores=[]
			while not done:
				t+=1
				q_k = sess.run(self.Qout, feed_dict = {self.state: state.reshape(1,28,28,4)})
				act = np.argmax(q_k)
				obs, R, done, _ = env.step(act)
				state = new_state(state, obs)
				scores.append( clip_reward(R) * 0.99 ** t)
				if done: 
					episode_lengths_list.append(t+1)
					episode_returns_list.append(np.sum(np.asarray(scores)))
					break
		return np.mean(np.array(episode_lengths_list)), np.mean(np.array(episode_returns_list))


def training_episode_with_replay_target(agent, target_agent, epsilon, env, test_env, sess, step_idx, update_operation):
	"""train using a replay buffer and target network for one episode of experience"""
	test_length = test_return = loss = 0
	obs, state = get_init_state(env)
	done = False

	while not done:
		step_idx += 1
		a, Qall = sess.run([agent.maxQ, agent.Qout], feed_dict={agent.state: state.reshape(-1, 28, 28, 4)})
		a = agent.epsilon_greedy(a, epsilon, env)
		next_state, R, done, _  = env.step(a)
		R = clip_reward(R)
		next_state = new_state(state, next_state)
		transition_exp = (state, a, R, next_state, int(not done))
		agent.experience_replay.remember_experience(transition_exp)

		if len(agent.experience_replay.replay_buffer) >= agent.experience_replay.buffer_size:
			sampled_experience = agent.experience_replay.sample(32)
			states_batch = np.stack(sampled_experience[:,0], axis=0)
			next_states_batch = np.stack(sampled_experience[:,3], axis=0)
			rewards_batch = np.stack(sampled_experience[:,2], axis=0)
			is_terminal_batch = np.stack(sampled_experience[:,-1], axis=0)
			actions_batch = sampled_experience[:,1].reshape(-1,1)
		
			l, _ = sess.run([agent.loss, agent.updateModel], feed_dict={agent.state: states_batch,
																	    target_agent.next_state: next_states_batch,
		                    									   	    agent.actions: actions_batch,
		                    									   	    agent.next_state: next_states_batch,
		                    									   	    agent.reward: rewards_batch.reshape(-1,1),
		                    									   	    agent.is_state_terminal: is_terminal_batch.reshape(-1,1)})
			loss += l
		state = next_state

		if step_idx % agent.update_target_frequency == 0:
			map(lambda x: sess.run(x), update_operation)
		if step_idx % agent.test_freq == 0:
			test_length, test_return = agent.evaluate(test_env, sess)
		if done:
			break
		break

	return step_idx, loss, test_length, test_return

def run_Q_learning(game):
	"""train and test a q-learning agent with replay"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run-var', default='')
	FLAGS, unparsed = parser.parse_known_args()		

	MODEL_DIR = "../trained_models"
	save_path = MODEL_DIR + '/' + str(game) 
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	PERF_DIR = "../performance_models"
	perf_path = PERF_DIR + '/' + str(game) 
	if not os.path.exists(perf_path):
		os.makedirs(perf_path)
	
	test_env = gym.make(game)
	env = gym.make(game)
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	# learning parameters
	print_frequency = 2
	learning_rate = 0.001
	epsilon = 0.1
	max_total_steps = 1000000

	tf.reset_default_graph()
	agent = DQNagent(learning_rate, state_size, action_size, 'main')
	target_agent = DQNagent(learning_rate, state_size, action_size, 'replay')
	agent.Build_Qlearning_graph(target_agent)
	tf_vars = tf.trainable_variables()
	update_operation = [tf_vars[ix+len(tf_vars)//2].assign(var.value()) for ix, var in enumerate(tf_vars[0:len(tf_vars)//2])]

	init = tf.global_variables_initializer()
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
	mean_lengths = []
	mean_returns = []

	with tf.Session() as sess:
		sess.run(init)

		if FLAGS.run_var == 're-run': 
			print('retraining and re-testing agent...')
			start_time = time.time()
			
			episodes_loss, episodes_len, episodes_return = [], [], []
			total_steps = e = 0
			# Derive an epsilon greedy policy from the action-value function
			while total_steps < max_total_steps:	
				e += 1
				state = env.reset()
				current_steps = total_steps
				total_steps, loss, test_length, test_return = training_episode_with_replay_target(agent, target_agent, epsilon, env, test_env, sess, total_steps, update_operation)
				t = total_steps-current_steps
				episodes_loss.append(loss/t)
				if e % print_frequency == 0:
					print("Time taken for episodes {} to {} was {:2f}s".format(max(0,e+1-print_frequency),e,time.time()-start_time))
					print('steps: ', total_steps)
					print("Train loss: ", (loss/(t+1)))
					np.save(perf_path + '/loss', episodes_loss)
					print("\n")
				if test_length != 0:
					episodes_len.append(test_length)
					episodes_return.append(test_return)
					print('test_len: ', test_length)
					print('test_return: ', test_return)
					np.save(perf_path + '/length', episodes_len)
					np.save(perf_path + '/return', episodes_return)
					
				saver.save(sess, save_path + '/' + str(game))

		else:
			print('re-evaluating on saved model')
			load_path  = save_path + '/' + str(game) + '.meta'
			num_repeats = 1
			saver = tf.train.import_meta_graph(load_path)
			saver.restore(sess, tf.train.latest_checkpoint(save_path)) 

			for t in range(num_repeats):
				sampled_length, sampled_return = agent.evaluate(test_env, sess)
				mean_lengths.append(sampled_length)
				mean_returns.append(sampled_return)

			print('-----------------------------')
			print('mean frame count: %f' % np.mean(mean_lengths))
			print('variance in survival length: %f' % np.var(mean_lengths))
			print('mean return: %f' % np.mean(mean_returns))
			print('variance in return: %f' % np.var(mean_returns))
			print('-----------------------------')

	sess.close()

def main():
	games = {'Pong-v3','MsPacman-v3','Boxing-v3'}
	for g in games:
		run_Q_learning(g)	

if __name__ == "__main__":
	main()