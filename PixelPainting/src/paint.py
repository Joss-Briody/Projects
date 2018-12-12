from __future__ import division
from __future__ import print_function
import sys, os
import argparse
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb
import time
import matplotlib.pyplot as plt

def get_images(np_x_initial, np_x_final, predictions_list, Xentropy_list, gt_Xentropy_list, n_predict, n_images, FLAGS):

	types = ["Good", "Bad", "High-Var", "Mistake"]
	Xent = np.stack(Xentropy_list,axis=1)
	meanXent = np.mean(Xent, axis=1)
	img_to_print = [np.argmin(np.mean(np.stack(gt_Xentropy_list,axis=1),axis=1)), np.argmax(meanXent), np.argmax(np.var(Xent, axis=1)) ]
	img_to_print = [16,60,42,54]

	k = 1
	if n_predict == 1:
		n_images = 1

	for ii in range(len(img_to_print)):
		img = img_to_print[ii]
		type_ = types[ii]
		real = 2*np.reshape(np.append(np_x_initial[img,:], np_x_final[img,:]), [28,28] )
		plt.subplot(len(img_to_print),n_images+2,k)
		plt.axis('off')
		plt.title('Original', fontsize=9.5)
		plt.imshow(real, cmap='gray', interpolation=None)
		k += 1

		for jj in range(n_images):
			pred_ = predictions_list[jj]
			prediction = np.append(np_x_initial[img,:], pred_[img, :])
			prediction = np.reshape(np.append(2*prediction-1,np.zeros([300-n_predict,1]) ), [28,28])
			plt.subplot(len(img_to_print),n_images+2,k)
			plt.axis('off')
			plt.title('%s %d' % (type_, jj+1), fontsize=9.5)
			plt.imshow(prediction, cmap='gray', interpolation='None')
			k += 1
		
		masked = np.reshape(np.append(2*np_x_initial[img,:]-1, np.zeros([300,1])),[28,28])
		plt.subplot(len(img_to_print),n_images+2,k)
		plt.axis('off')
		plt.title('Masked', fontsize=9.5)
		plt.imshow(masked, cmap='gray', interpolation='None')
		k += 1

	plt.savefig("../printed_pictures/" + "example_" + str(FLAGS.model) + "_" + str(FLAGS.num_units) + "_"  + str(n_predict) + ".png")
	# plt.show()


def load_data(FLAGS):
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

	return  mnist, mnist.train.images.shape[1]

def sample_images(test_set, n_mask):
	np.random.seed(420)
	n = np.random.randint(0, test_set.shape[0], size=100)
	masked = test_set[n,:test_set.shape[1]-n_mask]
	missing = test_set[n,test_set.shape[1]-n_mask:]
	return binarize(masked.reshape(100,-1,1)), binarize(missing.reshape(100,-1,1))


def binarize(images, threshold=0.1):
	return (threshold < images).astype('float32')


def rnn_net(initial_sequence, n_predict, n_RNN, n_layers):

	pred_probs = []
	weights = tf.Variable(tf.random_normal([n_RNN, 1]))
	bias =  tf.Variable(tf.random_normal([1]))

	GRU_cell = tf.nn.rnn_cell.GRUCell(num_units=n_RNN)

	if n_layers > 1:
		GRU_cell = tf.nn.rnn_cell.MultiRNNCell([GRU_cell]*n_layers)

	# get output and state for last visible pixel
	out, state = tf.nn.dynamic_rnn(cell=GRU_cell, inputs=initial_sequence, dtype=tf.float32) 
	out = tf.reshape(out, [-1, n_RNN])        # shape = [batch_size * (image_size * image_size), n_LSTM]
	out  = tf.matmul(out,weights) + bias       # shape = [batch_size * (image_size * image_size), 1]
	out  = tf.reshape(out, [-1, 484])
	last_out = out[:,-1]
	last_out_prob = tf.reshape(tf.nn.sigmoid(last_out),[100,1,1])
	pred_probs.append(tf.reshape(last_out_prob,[100,1]))

	# reuse the saved cell for the recursive prediction
	tf.get_variable_scope().reuse_variables()

	# predict on last 300 (or 28...) pixels recursively
	for _ in range(n_predict - 1):
		# out shape = [num_samples * batch, 1, n_RNN]
		output, state = tf.nn.dynamic_rnn(cell=GRU_cell, inputs=tf.round(last_out_prob), dtype=tf.float32, initial_state=state) 
		output = tf.reshape(output, [-1, n_RNN])
		output = tf.matmul(output, weights) + bias
		last_out_prob = tf.nn.sigmoid(tf.reshape(output, [100,1,1]))
		pred_probs.append(tf.reshape(last_out_prob, [100,1]))

	return pred_probs

def sample_output_dist(outputs, num_samples):
	# sample from the distribution of next pixel, given up to current pixels
	logits = tf.reshape(outputs, [-1,num_samples,1])

	# most likely - for prediction
	outputs_ml = logits[:,0,:]
	probs = tf.sigmoid(outputs_ml)
	ml_sample = tf.reshape(tf.round(probs), [-1,1,1])   # shape = [(num_samples-1) * batch, 1, 1]

	# bernoulli - for loss calculation
	outputs_bern = logits[:,1:,:]
	bern_sample = tf.contrib.distributions.Bernoulli(logits=outputs_bern, dtype=tf.float32)

	sample = tf.concat(1,[ml_sample, bern_sample.sample()])
		
	return tf.reshape(sample, [-1,1,1])

def compute_loss(logits, targets, num_samples):
	# logits: list of shape [no. to predict, samples * batchsize, 1] 
	# targets: ground truth images of shape [batchsize, no. to predict, 1]
		
	shape = [100,300]
	tar = tf.reshape(targets, [shape[0], shape[1]])
	tar = tf.tile(tar, [1,num_samples])
	tar = tf.reshape(tar, [-1, shape[1]])
	log = tf.stack(logits,axis=0) # log shape: [300,nsamples*nbsample,1]
	log = tf.reshape(tf.transpose(log,perm=[1,0,2]),[-1,300]) # log shape: [nsamples*nbsample,300]

	# entropy for each image
	sig_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=tar,logits=log),1)

	# loss is mean entropy
	loss = tf.reduce_mean(sig_entropy)

	# entropy for each image
	img_entropy = tf.reshape(sig_entropy, [shape[0], num_samples, -1])
	img_entropy = tf.reduce_mean(img_entropy,1)

	return loss, img_entropy

def print_Xent_cost(pred_probs, np_x_final, n_predict, FLAGS):
	n_samples = 10
	predictions_list = []
	Xentropy_list = []
	gt_Xentropy_list = []
	sum_Xentropy_list = []
		
	if n_predict > 1:
		for _ in range(n_samples):
			bin_sample = np.random.binomial(n=1,p=pred_probs)
			predictions_list.append(bin_sample)
			Xentropy = -np.sum(bin_sample*np.log(pred_probs) + (1-bin_sample)*np.log(1-pred_probs), axis=1)
			Xentropy_list.append(Xentropy)
			gt_Xent = -np.sum(np.reshape(np_x_final[:,:n_predict],[100,-1])*np.log(pred_probs) 
				+ (1-np.reshape(np_x_final[:,:n_predict],[100,-1]))*np.log(1-pred_probs), axis=1) 

			gt_Xentropy_list.append(gt_Xent) #-np.sum(np.reshape(np_x_final[:,:n_predict],[100,-1])*np.log(pred_probs), axis=1))
			sum_Xentropy_list.append(np.sum(Xentropy))
		
	else:
			ml_sample = np.round(pred_probs)
			ml_cost = -(ml_sample*np.log(pred_probs))
			sum_Xentropy_list.append(np.sum(ml_cost))
			Xentropy_list.append(ml_cost)
			gt_Xentropy_list.append(-np.sum( np.reshape(np_x_final[:,:n_predict],[100,-1])*np.log(pred_probs)
				+ (1-np.reshape(np_x_final[:,:n_predict],[100,-1]))*np.log(1-pred_probs), axis=1))
			predictions_list.append(ml_sample)
	
	np_x_final = np.reshape(np_x_final,[100,-1])
	gt_cost = -np.sum( np.reshape(np_x_final[:,:n_predict],[100,-1])*np.log(pred_probs) + (1-np.reshape(np_x_final[:,:n_predict], [100,-1]))*np.log(1-pred_probs) )/100
	mean_cost = sum(sum_Xentropy_list)/(100*len(sum_Xentropy_list))
	
	print("----------------------------------------------")
	print("Mean cost: ", mean_cost)
	print("Mean ground truth cost: ", gt_cost)
	print("----------------------------------------------")

	Xent = np.mean( np.asarray(Xentropy_list), axis=0 )
	gtXent = np.asarray(gt_Xentropy_list[0])

	plt.subplot(2,1,1)
	plt.hist(gtXent, bins=35)
	plt.xlabel('Ground truth cross entropy')
	plt.ylabel('Frequency')
	plt.title('Histograms of GT cross entropy and prediction cross entropy')
	plt.subplot(2,1,2)
	plt.hist(Xent, bins=35)
	plt.xlabel('Prediction cross entropy')
	plt.ylabel('Frequency')

	plt.tight_layout()
	plt.savefig("../printed_pictures/" + "Q2_XentHist_" + str(FLAGS.model) + "_" + str(FLAGS.num_units) + "_" + str(FLAGS.num_predict) + ".png")

	return Xentropy_list, gt_Xentropy_list, predictions_list

	######################## build the network to reload ########################

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='../data/', help='Directory for storing input data')
	parser.add_argument('--num_predict')
	parser.add_argument('--model')
	parser.add_argument('--num_units')
	FLAGS, unparsed = parser.parse_known_args()

	mnist, n_pixels = load_data(FLAGS)

	n_mask = 300
	n_pixels = 784
	n_hidden = 100
	n_images = 5

	if not FLAGS.num_units is None:
		n_RNN = int(FLAGS.num_units)

	n_layers = 1
	if FLAGS.model == 'b':
		n_RNN = 32
		n_layers = 3

	if not FLAGS.num_predict is None:
		n_predict = int(FLAGS.num_predict)
	else:
		# if user does not specify default to one row
		n_predict = 28

	# inputs
	x_initial = tf.placeholder(tf.float32, [None, n_pixels-300, 1])
	x_final = tf.placeholder(tf.float32, [None, 300, 1])

	pred_pixels = rnn_net(x_initial, n_predict, n_RNN, n_layers)

	# operation to initialise variables in graph
	init_op = tf.global_variables_initializer()

	# to save and reload a model
	saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)

	################################# reload trained state ###################################
	x_i, x_f = sample_images(mnist.test.images, 300)

	sp = "prediction_model_" + FLAGS.model + str(n_RNN) + "/"
	print_iters = 10
	with tf.Session() as sess:
		sess.run(init_op)
	
		# reload trained model from previous part of question
		print("Re-loading saved model for prediction...")
		load_path = "../saved_models/" + sp + "prediction_model_" + FLAGS.model + str(n_RNN) + ".meta"
		saver = tf.train.import_meta_graph(load_path)
		saver.restore(sess, tf.train.latest_checkpoint('../saved_models/'+ sp))
		
		[pred_probs, np_x_final, np_x_initial] = sess.run([pred_pixels, x_final, x_initial], feed_dict={x_initial: x_i, x_final: x_f} )

		sess.close()

	print("predicting...")
	pred_probs = np.stack(pred_probs, axis=1).reshape(100, n_predict)
	Xentropy_list, gt_Xentropy_list, predictions_list = print_Xent_cost(pred_probs, np_x_final, n_predict, FLAGS)

	print("producing and saving image examples...")
	get_images(np_x_initial, np_x_final, predictions_list, Xentropy_list, gt_Xentropy_list, n_predict, n_images, FLAGS)


if __name__ == '__main__':
	main()