from __future__ import division
from __future__ import print_function
import os, sys
import random
import time
import numpy as np
import argparse
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def load_data(FLAGS):
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	return  mnist, mnist.train.images.shape[1]

def binarize(images, threshold=0.1):
	""" Binarize images """
	return (threshold < images).astype('float32')

def rnn_net(x, y, n_classes, n_RNN, n_layers, batch_size, learning_rate):
	""" builds the many-to-one model trained and tested for this question """
	Weights = {
		'w1': tf.Variable(tf.random_normal([n_RNN, 100])),
		'w2': tf.Variable(tf.random_normal([100, n_classes])),
	}

	Bias = {
		'b1': tf.Variable(tf.random_normal([100])),
		'b2': tf.Variable(tf.random_normal([n_classes])),
	}

	# GRU layer
	gru_cell = tf.nn.rnn_cell.GRUCell(num_units=n_RNN)
	
	if n_layers > 1:
		g_cells = tf.nn.rnn_cell.MultiRNNCell([gru_cell]*n_layers)
	else:
		g_cells = gru_cell

	outputs, state = tf.nn.dynamic_rnn(cell=g_cells, dtype=tf.float32, inputs=x)

	# linear and non-linear layers
	last_output = outputs[:, -1, :]
	layer_1 = tf.nn.relu(tf.matmul(last_output, Weights['w1']) + Bias['b1'])
	layer_2 = tf.matmul(layer_1, Weights['w2']) + Bias['b2']
	output = tf.nn.softmax(layer_2)

	# final output
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer_2))
	acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), 'float'))
	# train operation
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	return output, cost, train_op, acc


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='../data/', help='Directory for storing input data')
	parser.add_argument('-r', '--run-var', default='')
	parser.add_argument('--model')
	parser.add_argument('--num_units')
	FLAGS, unparsed = parser.parse_known_args()
	mnist, n_pixels = load_data(FLAGS)

	# Hyperparams
	n_epochs = 60
	n_classes = 10
	b_size = 500
	n_RNN = int(FLAGS.num_units)
	n_layers = 1
	if FLAGS.model == 'd':
		n_layers = 3
	learning_rate = 0.0007

	# Inputs
	x = tf.placeholder(tf.float32, [None, n_pixels, 1])
	y = tf.placeholder(tf.float32)
	batch_size = tf.placeholder(tf.int32)

	# Build TF computation graph
	output, cost, opt, acc = rnn_net(x, y, n_classes, n_RNN, n_layers, batch_size, learning_rate)
	init_op = tf.global_variables_initializer()
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)

	sp = "model_" + FLAGS.model + str(n_RNN) + "/"

	with tf.Session() as sess:
		sess.run(init_op)
		acc_array = []
		test_array = []

		if FLAGS.run_var == 'rerun':
			print('rerunning model and resaving...')
			n_batches = int(mnist.train.num_examples/b_size)
			print_it = 10

			for epoch in range(n_epochs):
				t0 = time.time()
				epoch_cost = 0
				epoch_acc_tr = 0
				epoch_acc_te = 0
				print("Epoch no. %d of %d..." % (epoch+1, n_epochs))

				for i in range(n_batches):
					batch_x, batch_y = mnist.train.next_batch(b_size)
					batch_x = binarize(batch_x.reshape(b_size, n_pixels, 1))
					c, o, a = sess.run([cost, opt, acc], feed_dict={x: batch_x, y: batch_y, batch_size: b_size})
					epoch_cost += c
					epoch_acc_tr += a
					bx, by = mnist.test.next_batch(b_size)
					bx = binarize(bx.reshape(b_size, n_pixels, 1))
					epoch_acc_te += sess.run(acc, feed_dict={x: bx, y: by, batch_size: b_size})
					if i % print_it == 0:
						print('Trained for %d batches - cost for last batch = %f, accuracy = %f' % (i, c, a))

				print("------- Cost for epoch %d = %f - (training time %fs)" % (epoch+1, epoch_cost, time.time()-t0))
				acc_array.append(epoch_acc_tr/n_batches)
				test_array.append(epoch_acc_te/n_batches)

				save_path = "../saved_models/" + sp + "model_" + FLAGS.model + str(n_RNN) 
				save_path = saver.save(sess,save_path)
				np.savetxt('test_curve',np.asarray(acc_array),delimiter=',')
				np.savetxt('train_curve',np.asarray(test_array),delimiter=',')

		else: 
			print('reloading saved model...')

			load_path = "../saved_models/" + sp + "model_" + FLAGS.model + str(n_RNN) + ".meta"
			saver = tf.train.import_meta_graph(load_path)
			saver.restore(sess, tf.train.latest_checkpoint('../saved_models/'+ sp)) # model_' + FLAGS.model + str(n_LSTM) + '/'))

			all_vars = tf.get_collection('vars')
			for v in all_vars:
				sess.run(v)

			# test accuracy
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), 'float'))
			total = 0
			cross_ent = 0
			test_b_size = 1000

			for i in range(10):
				batch_x, batch_y = mnist.test.next_batch(test_b_size)
				batch_x = binarize(batch_x.reshape(test_b_size, n_pixels, 1))
				total += accuracy.eval(feed_dict={x: batch_x, y: batch_y,  batch_size: test_b_size})
				cross_ent += cost.eval(feed_dict={x: batch_x, y: batch_y,  batch_size: test_b_size})

			print("----------------------------------------------")
			print('Test accuracy: ', total/10)
			print('Test cross entropy: ',  cross_ent/10)
			print("----------------------------------------------")

			n_batches = int(mnist.train.num_examples/b_size)
			if FLAGS.run_var != 'reload-entire':
				n_batches = 2

			total = 0
			cross_ent = 0
			# train accuracy
			for i in range(n_batches):
				batch_x, batch_y = mnist.train.next_batch(test_b_size)
				batch_x = binarize(batch_x.reshape(-1, n_pixels, 1))
				total += accuracy.eval(feed_dict={x: batch_x, y: batch_y,  batch_size: test_b_size})
				cross_ent += cost.eval(feed_dict={x: batch_x, y: batch_y,  batch_size: test_b_size})

			print("----------------------------------------------")
			print('Train accuracy: ', total/n_batches)
			print('Test cross entropy: ', cross_ent/n_batches)
			print("----------------------------------------------")

		sess.close()

if __name__ == "__main__":
	random.seed(0)
	main()
