from __future__ import division
from collections import deque
import gym
from gym import wrappers
import numpy as np
import random
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
import pdb
import time

def clip_reward(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else: 
		return 0
	

def preprocess_state(frame):
	resized = resize(frame, (28,28))
	return rgb2gray(resized)#.astype(np.uint8)

def stack_states(env):
	state = np.empty([28,28,4])
	obs = env.reset()
	for i in range(4):
		state[:,:,i] = preprocess_state(obs)
		action = env.action_space.sample()
		obs, _, _, _ = env.step(action)
	return state

def sample_environment(env, num_episodes, discount_factor):
	score_history = []
	length_history = []
	DATA = []

	for i in range(num_episodes):
		obs = env.reset()
		done = False
		scores = []
		frame_counts = []
		t=0
		while not done:
			t+=1
			action = env.action_space.sample()
			obs, R, done, _ = env.step(action)
			scores.append( clip_reward(R) * 0.99 ** t)
			if done: 
				score_history.append(np.sum(scores))
				length_history.append(t)
				break
	
	score_history = np.asarray(score_history)
	length_history = np.asarray(length_history)
	print '-------------------------------------------'
	print "mean score: ", np.mean(score_history) 
	print 'score std-dev: ', np.std(score_history)
	print '-------------------------------------------'
	print 'mean frame count: ', np.mean(length_history)
	print 'frame count std-dev: ', np.std(length_history)
	print '-------------------------------------------'


def main():
	
	games = {'Pong-v3','MsPacman-v3','Boxing-v3'}
	games = {'Boxing-v3'}
	
	discount_factor = 0.99
	num_episodes = 100
	for g in games:
		t0 = time.time()
		env = gym.make(g)
		print(g)
		sample_environment(env, num_episodes, discount_factor)
		print 'time taken: ', time.time() - t0

if __name__ == '__main__':
	main()