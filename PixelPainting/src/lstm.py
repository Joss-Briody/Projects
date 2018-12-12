from __future__ import division
import sys, os
import argparse
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time

def load_data(FLAGS):
	""" function to load the mnist data set """
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	return  mnist, mnist.train.images.shape[1]

def train(FLAGS):

	# set the learning hyperparameters
	mnist, n_pixels = load_data(FLAGS)
	n_classes = 10
	n_pixels = 784
	learning_rate = 0.001
	n_hidden = 100
	num_epochs = 60
	drop_out_keep_prob = 0.6
	b_size = 512

	# inputs
	x  = tf.placeholder(tf.float32, [None, n_pixels, 1])
	y = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)
	is_train = True

	def binarize(images, threshold=0.1):
		""" binarize the input data """
		return (threshold < images).astype('float32')

	def feed_dict(train, batchSize=b_size):
		""" function to map values onto tensorflow placeholders """
		if train:
			xs, ys = mnist.train.next_batch(batchSize)
			keepProb = drop_out_keep_prob
		else:
			num_to_test = 1012
			xs, ys = mnist.test.images, mnist.test.labels
			xs = xs[:num_to_test,:]
			ys = ys[:num_to_test]
			keepProb = 1.0
		
		xs = xs.reshape(-1, n_pixels, 1)
		return {x: binarize(xs), y: ys, keep_prob: keepProb}

	def weight_init(shape):
		""" function to initialise weights """
		init = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(init)

	def bias_init(shape):
		""" function to initialise bias variable """
		init = tf.constant(0.1, shape=shape)
		return tf.Variable(init)
	
	def linear_nn_layer(input_tensor, input_dim, output_dim):
		""" function to apply linear neural network layer """
		W = weight_init([input_dim,output_dim])	
		b = bias_init([output_dim])
		return tf.add( tf.matmul(input_tensor,W), b)


	def rnn_lstm(x, keep_prob, n_layers):
		""" to run an lstm cell """
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = n_RNN, state_is_tuple=True)
		if n_layers > 1:
			lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * n_layers)

		output, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype = tf.float32)
		return output[:,-1,:]

	def rnn_GRU(x, keep_prob, n_layers):
		""" to run a GRU cell """
		GRU_cell = tf.nn.rnn_cell.GRUCell(num_units = n_RNN)
		GRU_cell = tf.nn.rnn_cell.DropoutWrapper(GRU_cell,
                        input_keep_prob = 1, output_keep_prob = keep_prob)

		if n_layers > 1:
			GRU_cell = tf.nn.rnn_cell.MultiRNNCell([GRU_cell] * n_layers)

		output, states = tf.nn.dynamic_rnn(GRU_cell, x, dtype = tf.float32)
		return output[:,-1,:]

	def batch_norm(x, n_RNN, isTraining):
		""" function to apply batch normalisation (to output of RNN) """
		beta = tf.get_variable(name="beta",shape=[n_RNN], dtype = tf.float32,
			initializer=tf.constant_initializer(0.0),)
		gamma = tf.get_variable(name="gamma",shape=[n_RNN], dtype = tf.float32,
			initializer=tf.constant_initializer(0.0))
		batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		# for use with conditional batchnorm as arguments to cond 
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		def mean_var_no_update():
			return ema.average(batch_mean), ema.average(batch_var)

		# mean, var = tf.cond(tf.less(tf.constant(0, dtype=tf.float32),isTraining), mean_var_with_update, mean_var_no_update)
		mean, var = mean_var_with_update()
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
		return normed
	
	def rnn_net(x,y,kp,n_RNN,n_classes,n_hidden,cell_type,rate, is_train):
		""" builds the many-to-one model trained and tested for this question """

		Weights = {
			'w1': tf.Variable(tf.random_normal([n_RNN,n_hidden])),
			'w2': tf.Variable(tf.random_normal([n_hidden,n_classes])),
		}

		Bias = {
			'b1': tf.Variable(tf.random_normal([n_hidden])),
			'b2': tf.Variable(tf.random_normal([n_classes])),
		} 
	
		cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_RNN,state_is_tuple=True)
		cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=1,output_keep_prob=kp)

		if n_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([cell]*n_layers)
		
		outputs, states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32,inputs=x)
		last_output = outputs[:,-1,:]
		
		if cell_type == 'lstm':
			last_output_normed = batch_norm(last_output, n_RNN, is_train)

		elif cell_type == 'GRU':
			last_output_normed = last_output

		layer_1 = tf.nn.relu(tf.matmul(last_output,Weights['w1'])+Bias['b1'])
		layer_2 = tf.matmul(layer_1,Weights['w2'])+Bias['b2']
	
		output = tf.nn.softmax(layer_2)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_2,labels=y))
		
		global_step = tf.Variable(0,trainable=False)
		rate_ = tf.train.exponential_decay(rate, global_step, 2*mnist.train.images.shape[0]/b_size,0.99,staircase=False)
		train_op = tf.train.AdamOptimizer(rate_).minimize(cost,global_step=global_step)

		return output, cost, train_op, rate_

	######################## build the network for each question ########################

	if not FLAGS.num_units is None:
		n_RNN = int(FLAGS.num_units)

	n_layers = 1
	if FLAGS.model == 'c':
		n_RNN = 32
		n_layers = 3

	cell_type = 'lstm'

	if FLAGS.model not in ['a','c']:
		raise ValueError('invalid model entered for this script!')

	# build the tf computation graph
	output, cost, train_op, decayed_rate = rnn_net(x,y,keep_prob,n_RNN,n_classes,n_hidden,cell_type,learning_rate, is_train)
	
	########################## build the tf computation graph ###########################

	correct_pred = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# operation to initialise variables in graph
	init_op = tf.global_variables_initializer()

	# to save and reload a model
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)

	################################# run the session ###################################
	
	print_iters = 100
	sp = "model_" + FLAGS.model + str(n_RNN) + "/"
	acc_array = []
	test_array = []
	t0 = time.time()
	with tf.Session() as sess:
		sess.run(init_op)

		# retrain the model. WARNING: IF YOU CHOOSE THIS IT WILL OVERWRITE THE SAVED MODEL
		if FLAGS.run_var == 'rerun':
			print('rerunning model and resaving...')
			for i in range(int(num_epochs*mnist.train.images.shape[0]/b_size)):
				c,o = sess.run([cost,train_op], feed_dict = feed_dict(True))
				if i % print_iters == 0:
					acc = sess.run(accuracy, feed_dict = feed_dict(True))
					print("iter %3d -- Training Accuracy: %f"%(i,acc))
					acc_array.append(acc)
					test_acc = sess.run(accuracy, feed_dict=feed_dict(False))
					test_array.append(test_acc)
					print('test_acc: %f'%test_acc)	
				if i % 50 == 0:
					print('time: '+str(time.time()-t0))
					t0 = time.time()
					if cell_type == 'LSTM':
						print('rate: ')
						print(sess.run(decayed_rate))
				
				if i*1000 % (3*mnist.train.images.shape[0]) == 0 or i==int(num_epochs*mnist.train.images.shape[0]/b_size)-1:
					save_path = "../saved_models/" + sp + "model_" + FLAGS.model + str(n_RNN) 
					save_path = saver.save(sess,save_path)
					np.savetxt('test_curve',np.asarray(acc_array),delimiter=',')
					np.savetxt('train_curve',np.asarray(test_array),delimiter=',')
		
		# reload the saved model
		else: 
			print('reloading saved model...')
			load_path = "../saved_models/" + sp + "model_" + FLAGS.model + str(n_RNN) + ".meta"
			saver = tf.train.import_meta_graph(load_path)
			saver.restore(sess, tf.train.latest_checkpoint('../saved_models/'+ sp)) 
			all_vars = tf.get_collection('vars')
			for v in all_vars:
				sess.run(v)
		
		total_test_acc = []
		total_train_acc = []
		test_S = []
		train_S = []

		# calculate the final training and test error
		test_b_size = 1000
		for idx in range(int(mnist.test.images.shape[0]/test_b_size)):
			x_ = mnist.test.images[idx*test_b_size:(idx+1)*test_b_size,:]
			y_ = mnist.test.labels[idx*test_b_size:(idx+1)*test_b_size]
			x_ = x_.reshape(-1, n_pixels, 1)
			total_test_acc.append(sess.run(accuracy, feed_dict={x: binarize(x_), y: y_, keep_prob:1}))
			test_S.append(sess.run(cost, feed_dict = {x: binarize(x_), y: y_, keep_prob:1}))
		
		num_batches = int(mnist.train.images.shape[0]/test_b_size)

		# the reported train accuracies have been calculated on the entire training set,
		# however this takes >10 mins to run on a laptop so you might not want to do that
		if FLAGS.run_var != 'reload-entire':
			num_batches = 2

		for idx in range(num_batches):
			x_ = mnist.train.images[idx*test_b_size:(idx+1)*test_b_size,:]
			y_ = mnist.train.labels[idx*test_b_size:(idx+1)*test_b_size]
			x_ = x_.reshape(-1, n_pixels, 1)
			total_train_acc.append(sess.run(accuracy, feed_dict={x: binarize(x_), y: y_, keep_prob:1}))
			train_S.append(sess.run(cost, feed_dict = {x: binarize(x_), y: y_, keep_prob:1}))

		print("----------------------------------------------")
		print( "Final test accuracy: %f" % (np.mean(np.asarray(total_test_acc) )))
		print(  "Final train accuracy: %f" % (np.mean(np.asarray(total_train_acc)))) 
		print("----------------------------------------------")
		print("----------------------------------------------")
		print(" Test Cross Entropy: %f" %(np.mean(np.asarray(test_S))))
		print(" Train Cross Entropy: %f" %(np.mean(np.asarray(train_S))))		
		print("----------------------------------------------")
		sess.close()

	####################################################################################

def main():
	# user defined arguments to specify which model and whether or not to retrain
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='../data/',
		help='Directory for storing input data')
	parser.add_argument('-r', '--run-var', default='')
	parser.add_argument('--model')
	parser.add_argument('--num_units')
	FLAGS, unparsed = parser.parse_known_args()
	
	train(FLAGS)

if __name__ == '__main__':
	main()
