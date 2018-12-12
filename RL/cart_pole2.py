from __future__ import division
from collections import deque
import gym
from gym import wrappers
import numpy as np
import random
import tensorflow as tf
import pdb
import time

def sample_environment(env, max_episode_length, num_episodes, discount_factor):
	t0 = time.time()
	score_history = []
	length_history = []
	DATA = []
	for i in range(num_episodes):
		obs = env.reset()
		done = False
		states = []
		actions = []
		rewards = []
		for j in range(max_episode_length):
			data = np.zeros(6)
			data[:4] = obs
			action = env.action_space.sample()
			data[4] = action
			obs, _, done, _ = env.step(action)
			if done: 
				data[5] = -1
				DATA.append(data)
				score_history.append(-1 * discount_factor ** (j-1) )
				length_history.append(j)
				break
			else:
				data[5] = 0
				DATA.append(data)
			
	DATA = np.asarray(DATA)
	score_history = np.asarray(score_history)
	length_history = np.asarray(length_history)
	print 'time taken: ', time.time() - t0

	print '-------------------------------------------'
	print "mean score: ", np.mean(score_history) 
	print 'score variance: ', np.var(score_history)
	print '-------------------------------------------'
	print 'mean episode length: ', np.mean(length_history)
	print 'episode length variance: ', np.var(length_history)
	print '-------------------------------------------'
	
	return DATA

def main():
	discount_factor = 0.99
	max_episode_length = 300
	num_episodes = 2000
	
	env = gym.make('CartPole-v0')
	DATA = sample_environment(env, max_episode_length, num_episodes, discount_factor)
	np.save('sampled_experience' + str(max_episode_length) + '_episodes_.npy', DATA)

if __name__ == '__main__':
	main()