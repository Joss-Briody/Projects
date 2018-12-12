from __future__ import division
from __future__ import print_function
import sys, os
import argparse
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb
import time

def load_data(FLAGS):
	""" function to load the mnist data set """
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	return  mnist, mnist.train.images.shape[1]

def binarize(images, threshold=0.1):
	""" binarize the input data """
	return (threshold < images).astype('float32')

def rnn_net(x, n_RNN, n_layers, n_pixels, learning_rate):
	""" builds the many-to-many model trained and tested for this question """

	weights = tf.Variable(tf.random_normal([n_RNN, 1]))
	bias =  tf.Variable(tf.random_normal([1]))

	GRU_cell = tf.nn.rnn_cell.GRUCell(num_units=n_RNN)

	if n_layers > 1:
		GRU_cell = tf.nn.rnn_cell.MultiRNNCell([GRU_cell]*n_layers)

	out, state = tf.nn.dynamic_rnn(cell=GRU_cell, inputs=x, dtype=tf.float32) 
	out = tf.reshape(out, [-1, n_RNN])        # shape = [batch_size * (image_size * image_size), n_LSTM]
	out  = tf.matmul(out,weights) + bias       # shape = [batch_size * (image_size * image_size), 1]
	out  = tf.reshape(out, [-1, n_pixels])
	pred  = tf.nn.sigmoid(out)
	x_reshaped = tf.reshape(x, [-1, n_pixels])        # shape = [batch_size, (image_size * image_size)]

	acc_ = tf.cast(tf.equal(tf.round(tf.sigmoid(out[:,:-1])), x_reshaped[:,1:]), 'float')
	diff = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(targets=x_reshaped[:,1:], logits=out[:,:-1]), reduction_indices=[1] )
	Xentropy = tf.reduce_mean(diff)
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Xentropy)

	return Xentropy, train_op, acc_

	######################## build the network to reload ########################

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='../data/', help='Directory for storing input data')
	parser.add_argument('-r', '--run-var', default='')
	parser.add_argument('--model')
	parser.add_argument('--num_units')
	FLAGS, unparsed = parser.parse_known_args()

	mnist, n_pixels = load_data(FLAGS)

	n_epochs = 30
	n_pixels = 784
	learning_rate = 0.001
	b_size = 500

	if not FLAGS.num_units is None:
		n_RNN = int(FLAGS.num_units)

	n_layers = 1
	if FLAGS.model == 'b':
		n_RNN = 32
		n_layers = 3

	########################## build the tf computation graph ################################

	x = tf.placeholder(tf.float32, [None, n_pixels, 1])
	Xent, train_op, acc_ = rnn_net(x, n_RNN, n_layers, n_pixels, learning_rate)

	# operation to initialise variables in graph
	init_op = tf.global_variables_initializer()

	# to save and reload a model
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)

	################################# reload trained state ###################################

	sp = "prediction_model_" + FLAGS.model + str(n_RNN) + "/"
	print_iters = 10
	with tf.Session() as sess:
		sess.run(init_op)
		cost_array = []
		cost_array_test = []
		
		if FLAGS.run_var == "rerun":
			print("Re-running model (and re-saving)...")
			n_batches = int(mnist.train.num_examples/b_size)

			for epoch in range(n_epochs):
				t0 = time.time()
				epoch_cost = 0
				epoch_cost_test = 0

				print("Epoch no. %d of %d..." % (epoch+1, n_epochs))

				for i in range(n_batches):
					xs, ys = mnist.train.next_batch(b_size)
					xs = xs.reshape(-1, n_pixels, 1)

					cost, op = sess.run([Xent,train_op], feed_dict = {x: binarize(xs) })
					epoch_cost += cost

					xs_, ys_ = mnist.test.next_batch(b_size)
					xs_  = xs_.reshape(-1, n_pixels, 1)

					cost_test = sess.run(Xent, feed_dict={x: binarize(xs_) })
					epoch_cost_test += cost_test

					if i % print_iters == 0:
						print('Trained for %d batches - cost for last batch = %f' % (i, cost))

				print("------- Cost for epoch %d = %f - (training time %fs)" % (epoch+1, epoch_cost, time.time()-t0))
				cost_array.append(epoch_cost)
				cost_array_test.append(epoch_cost_test)

				save_path = "../saved_models/" + sp + "prediction_model_" + FLAGS.model + str(n_RNN) 
				save_path = saver.save(sess,save_path)

				np.savetxt('test_cost',np.asarray(cost_array_test),delimiter=',')
				np.savetxt('train_cost',np.asarray(cost_array),delimiter=',')
		

		else:
			print("Re-loading saved model...")

			load_path = "../saved_models/" + sp + "prediction_model_" + FLAGS.model + str(n_RNN) + ".meta"
			saver = tf.train.import_meta_graph(load_path)
			saver.restore(sess, tf.train.latest_checkpoint('../saved_models/'+ sp)) 

			all_vars = tf.get_collection('vars')
			for v in all_vars:
				sess.run(v)

			xs, ys = mnist.train.next_batch(b_size)
			xs = xs.reshape(-1, n_pixels, 1)

			xs_, ys_ = mnist.test.next_batch(b_size)
			xs_  = xs_.reshape(-1, n_pixels, 1)

			# calculate the final training and test losses
			print("----------------------------------------------")
			print( "Final training loss: %f" % ( sess.run(Xent, feed_dict={x: binarize(xs) })) )
			print(  "Final testing loss: %f" % ( sess.run(Xent, feed_dict={x: binarize(xs_) })) )
			print("----------------------------------------------")
			print( "Final training accuracy: %f" % ( sess.run(tf.reduce_mean(acc_), feed_dict={x: binarize(xs) })) )
			print(  "Final testing accuracy: %f" % ( sess.run(tf.reduce_mean(acc_), feed_dict={x: binarize(xs_) })) )
			print("----------------------------------------------")		

		sess.close()


if __name__ == '__main__':
	main()
