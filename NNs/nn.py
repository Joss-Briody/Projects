import numpy as np
import sys, os
import argparse
import time
import pickle

from layers import *


class NeuralNetwork:
	"""contains the individual units in a chain"""
	def __init__(self, M, rate):
		self.learning_rate = rate
		self.batch_size = M
		self.layers = []
		self.numLayers = 0

	def add_layer(self, layer):
		self.layers.append(layer)
		self.numLayers += 1

	def forward_all(self, x):
		"""string together forward pass for each module"""
		for l in range(0,self.numLayers):
			y = self.layers[l].forward_pass(x)
			x = y
		# the last y in the chain ouputs class probabilities
		return y

	def backward_all(self, dL_dy, y):
		"""string together backwards passes for each module - if module has 
			parameters, update them"""
		for l in range(self.numLayers-1,-1,-1):
			dL_dx = self.layers[l].backward_pass(dL_dy, y)
			if callable(getattr(self.layers[l],"param_gradient", None)):
				dW, db = self.layers[l].param_gradient(dL_dy, y)
				self.layers[l].update_params(self.learning_rate, dW, db)
			dL_dy = dL_dx
			y = self.layers[l].x

	def train(self, train_images, train_labels, num_epochs, num_examples,
	 		  test_images, test_labels, numPixels, numClasses):
		"""train neural network using batch gradient descent
		"""
		test_error = []
		train_error = []
		for j in range(0, num_epochs):
			for i in range(0, num_examples, self.batch_size):
				if i%4000 == 0:
					print("Iter-number: %d" % i)
					test_error.append(self.test(test_images, test_labels, numPixels, numClasses))
					train_error.append(self.test(train_images, train_labels, numPixels, numClasses))
				x = train_images[i:i+self.batch_size,:]
				y = train_labels[i:i+self.batch_size,:]
				# the last module in the chain does the evaluation - give it the correct label
				self.layers[self.numLayers-1].set_true_label(y)
				loss = self.forward_all(x)
				dL_dy = 1
				self.backward_all(dL_dy,1)
			# permute data inbetween each epoch	
			p = np.random.permutation(num_examples)
			train_images = train_images[p,:]
			train_labels = train_labels[p,:]

		return test_error, train_error


	def test(self, test_images, test_labels, numPixels, numClasses):
		"""computes the test error of the network"""
		self.err=0
		
		for i in range(0, test_labels.shape[0]):
			x_test = test_images[i,:].reshape(1,1,28,28)
			y_test = test_labels[i,:].reshape(1,numClasses)
			y_pred = self.forward_all(x_test)
			if np.argmax(y_pred) != np.argmax(y_test):
				self.err+=1

		err = float(self.err)/test_labels.shape[0]
		return err


def run_all(FLAGS):

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

	train_images = mnist.train.images
	train_labels = mnist.train.labels
	test_images  = mnist.test.images
	test_labels  = mnist.test.labels

	numPixels = train_images.shape[1]

	train_images = np.reshape(train_images,(train_images.shape[0],1,28,28))
	test_images  = np.reshape(test_images,(test_images.shape[0],1,28,28))

	numUnits = 256
	numClasses = 10
	numEpochs = 5
	batchSize = 50
	num_examples = train_labels.shape[0]

	if FLAGS.run_var == 'rerun':
		print("Re-running model (and saving)...")
		nn = neural_network( batchSize, 0.1)

		# build the network corresponding to model d
		nn.add_layer( convolutional_layer([16,1,3,3], 1) )
		nn.add_layer( relu_layer() )
		nn.add_layer( max_pool_layer(2,2) )
		nn.add_layer( convolutional_layer([16,16,3,3], 1) )
		nn.add_layer( relu_layer() )
		nn.add_layer( max_pool_layer(2,2) )
		nn.add_layer( flatten_layer(batchSize) )
		nn.add_layer( linear_layer([28*28, numUnits], [1,numUnits]) )
		nn.add_layer( relu_layer() )
		nn.add_layer( linear_layer([numUnits, numClasses], [1,numClasses]) )
		nn.add_layer( cross_entropy_logit_layer() )

		t0 = time.time()
		test_error_vec, train_error_vec = nn.train(train_images,train_labels, numEpochs, 
												   num_examples, test_images, test_labels,
												   numPixels, numClasses)
		t1 = time.time()
		print('training time: ' + str(t1-t0))

		# save neural network object with pickle
		with open("model_d/neural_net_d_object.pkl", "wb") as F:
			pickle.dump(nn, F, protocol = pickle.HIGHEST_PROTOCOL)
	else:
		print("Reloading trained model...")
		with open("model_d/neural_net_d_object.pkl", "rb") as F:
			nn = pickle.load(F)
		
	test_error = nn.test(test_images, test_labels, numPixels, numClasses)
	train_error = nn.test(train_images, train_labels, numPixels, numClasses)

	print('final test error = ' + str(test_error))
	print('final train error = ' + str(train_error))

	confusionMat = confusion_matrix(nn, test_images, test_labels, numPixels, numClasses)
	print(confusionMat)
	

def confusion_matrix(nn, test_images, test_labels, numPixels, numClasses):
	"""compute the confusion matrix"""
	confusionMat = np.zeros([numClasses, numClasses])
	for i in range(0, test_labels.shape[0]):
		x_test = test_images[i,:].reshape(1,1,28,28)
		y_test = int(np.argmax(test_labels[i,:].reshape(1,numClasses)))
		y_pred = int(np.argmax(nn.forward_all(x_test)))
		confusionMat[y_pred,y_test] += 1
	return confusionMat.astype(int)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--run-var', default='')
	FLAGS, unparsed = parser.parse_known_args()
	run_all(FLAGS)

if __name__ == '__main__':
	main()