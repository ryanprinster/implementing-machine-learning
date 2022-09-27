
import sys
print(sys.executable)
# sys.path.insert(0, '/Users/ryanprinster/Desktop')
# import wiper

import numpy as np

class fully_connected():
	def __init__(self, input_dim, output_dim, batch_size, activation_func):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.batch_size = batch_size
		self.W = np.random.rand(self.batch_size, self.input_dim, self.output_dim)
		self.b = np.random.rand(self.batch_size, 1, self.output_dim)
		self.activation_func = activation_func

		# Cache for access during backprop
		self.Z = None
		self.a = None

	def forward(self, X):
		pass

	def print_weights(self):
		print(self.W)

fully_connected(5, 2, 7)
