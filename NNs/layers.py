import numpy as np


def im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
	""" function to return indices for im2col indices
	"""
	N, C, H, W = x_shape

	out_height = int((H + 2 * padding - field_height) / stride + 1)
	out_width  = int((W + 2 * padding - field_width) / stride + 1)

	i0 = np.repeat(np.arange(field_height), field_width)
	i0 = np.tile(i0, C)
	i1 = stride * np.repeat(np.arange(out_height), out_width)
	j0 = np.tile(np.arange(field_width), field_height * C)
	j1 = stride * np.tile(np.arange(out_width), out_height)
	i = i0.reshape(-1, 1) + i1.reshape(1, -1)
	j = j0.reshape(-1, 1) + j1.reshape(1, -1)

	k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

	return (k, i, j)


def im2col(x, field_height, field_width, padding=1, stride=1):
	""" im2col function using np indexing and zero-padding to preserve input size
	"""
	x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
	
	# apply im2col
	k, i, j = im2col_indices(x.shape, field_height, field_width, padding, stride)
	cols = x_padded[:, k, i, j]
	cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * x.shape[1], -1)

	return cols


def col2im(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
	"""function to invert im2col operation as required in backwards pass
	"""
	N, C, H, W = x_shape
	H_padded = int(H + 2 * padding)
	W_padded = int(W + 2 * padding)
	x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

	k, i, j = im2col_indices(x_shape, field_height, field_width, padding, stride)
	cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
	cols_reshaped = cols_reshaped.transpose(2, 0, 1)
	np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

	if padding == 0:
		return x_padded

	return x_padded[:, :, padding:-padding, padding:-padding]

	
class LinearLayer:
	"""Linear network unit
	"""
	def __init__(self, shape, length):
		self.weights = np.random.normal(0, 0.1, shape)
		self.bias = 0.1*np.ones(length) 
	
	def forward_pass(self, x):
		self.x = x
		return np.dot(x, self.weights) + self.bias

	def backward_pass(self, dL_dy, y):
		return np.dot(dL_dy, self.weights.T)

	def param_gradient(self, dL_dy, y):
		x = self.x
		tmp = np.mean(dL_dy, axis=0, dtype=np.float64)
		return np.dot(x.T, dL_dy)/x.shape[0] , tmp.reshape(1,len(tmp))

	def update_params(self, rate, dW, db):
		"""update parameters via batch gradient descent
		"""
		self.weights += -rate * dW
		self.bias += -rate * db


class ReluLayer:
	"""ReLU activation function
	"""
	def forward_pass(self,x):
		self.x = x
		return np.maximum(x,0,x)

	def backward_pass(self, dL_dy, y):
		return np.multiply(dL_dy, (y > 0) )


class SoftMaxLayer:
	"""returns the softmax of the input"""
	def forward_pass(self, x):
		self.x = x
		# prevent overflow
		e = np.exp(x - np.max(x)) 
		return e / np.sum(e, axis=0) 
		
	def backward_pass(self, dL_dy,y):
		# this function is never actually called with these models
		return np.multiply(dL_dy,y) - np.dot(dL_dy,y) * y


class CrossEntropyLogitLayer:
	"""computes the cross entropy loss (objective function) taking as inputs the
		inputs for softmax. Currently returns vector y which is the softmax 
		of the inputs rather than the loss (i.e. scalar y) as we do not need to 
		explicitly  to compute the loss, just its derivatives in forward_pass() 
		and backward_pass()
	"""

	def set_true_label(self, label):
		self.true_label = label

	def forward_pass(self,x):
		self.x = x
		# prevent overflow
		e = np.exp(x - np.max(x,axis=1).reshape(x.shape[0],1))  
		e = np.divide( e , np.sum(e, axis=1).reshape(e.shape[0],1) )
		# store this for use in the backward pass
		self.softmax_x = e
		return e 

	def backward_pass(self, dL_dy, y):
		return self.softmax_x - self.true_label


class ConvolutionalLayer:
	"""convolutional network unit
	""" 
	def __init__(self, filter_shape, stride):
		self.stride = stride
		self.W_conv = np.random.normal(loc=0.0, scale=0.1, size=filter_shape)
		self.b_conv = 0.1*np.ones(([1,filter_shape[0],1,1]))
		self.stride = 1
		self.padding = 1

	def forward_pass(self, X):
		"""returns the convolutional of the 4d tensor x using im2col technique
		"""
		self.x = X
		n_filters, d_filter, h_filter, w_filter = self.W_conv.shape
		n_x, d_x, h_x, w_x = X.shape
		h_out = int( (h_x - h_filter + 2 * self.padding) / self.stride + 1)
		w_out = int( (w_x - w_filter + 2 * self.padding) / self.stride + 1)

		# save this for use in backwards pass
		self.X_col = im2col(X, h_filter, w_filter, padding=self.padding, stride=self.stride)

		W_col = self.W_conv.reshape(n_filters, -1)
		
		out = np.dot(W_col, self.X_col)
		out = out.reshape(n_filters, h_out, w_out, n_x)
		out = out.transpose(3, 0, 1, 2) + self.b_conv
		
		return out

	def backward_pass(self, dL_dy, y):
		"""the backwards pass is also a (rotated) convolution
		"""
		n_filter, d_filter, h_filter, w_filter = self.W_conv.shape
		
		dL_dy_reshaped = dL_dy.transpose(1, 2, 3, 0).reshape(n_filter, -1)
		
		W_reshape = self.W_conv.reshape(n_filter, -1)
		dX_col = np.dot(W_reshape.T, dL_dy_reshaped)
		dL_dx = col2im(dX_col, self.x.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)
		
		return dL_dx 

	def param_gradient(self, dL_dy, y):
		n_filter = self.W_conv.shape[0]
		db = np.sum(dL_dy, axis=(0, 2, 3))
		db = db.reshape(-1,n_filter, 1,1) / self.x.shape[0]
		dL_dy_reshaped = dL_dy.transpose(1, 2, 3, 0).reshape(n_filter, -1)
		dW = np.dot(dL_dy_reshaped, self.X_col.T)
		dW = dW.reshape(self.W_conv.shape) / self.x.shape[0]
		return dW, db

	def update_params(self, learning_rate, dW, db):
		self.W_conv += - learning_rate * dW
		self.b_conv += - learning_rate * db


class MaxPoolLayer:
	"""convolutional network unit: forward pass reduces image size by 2x2""" 
	def __init__(self, size, stride):
		self.pool_size = size
		self.stride = stride

	def forward_pass(self, X):
		"""Save the indexes of inputs which are allowed through in
		   forward pass in order to reconstruct image in the backwards pass
		""" 
		self.x = X
		n,d,h,w = X.shape
		h_out = int( (h - self.pool_size) / self.stride + 1 )
		w_out = int( (w - self.pool_size) / self.stride + 1 )
		self.input_shape = X.shape
		self.output_shape = [h_out, w_out]

		X_ = X.reshape(n*d,1,h,w)

		# save the indexes for max values to use in the backwards step
		self.X_col = im2col(X_, self.pool_size, self.pool_size, padding=0, stride=self.stride)
		self.max_idx = np.argmax(self.X_col, axis=0)

		out = self.X_col[self.max_idx, range(self.max_idx.size)]

		out = out.reshape(h_out, w_out, n, d)
		out = out.transpose(2, 3, 0, 1)
	
		return out

	def backward_pass(self, dL_dy, y):
		dX_col = np.zeros_like(self.X_col)
		dL_dy_flat = dL_dy.transpose(2,3,0,1).ravel()
		dX_col[self.max_idx, range(len(self.max_idx))] = dL_dy_flat
		
		dX = col2im(dX_col, (self.input_shape[0]*self.input_shape[1], 1, 
							 self.input_shape[2], self.input_shape[3]),
					self.pool_size, self.pool_size, padding=0, stride=self.stride)

		dX = dX.reshape(self.input_shape)
		return dX

class FlattenLayer:
	"""forward: 4d -> 2d, backward 2d -> 4d
	"""
	def __init__(self, batch_size):
		self.batch_size = batch_size
	
	def forward_pass(self, X):
		self.x = X
		return np.reshape(X,(-1,16*7*7) )

	def backward_pass(self, dL_dy, y):
		return np.reshape(dL_dy,(self.batch_size,16,7,7))