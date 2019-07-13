#!/usr/bin/env python3
import numpy as np

sd=394#np.random.randint(500)
print(sd)
np.random.seed(sd)

class neural_net:
	def __init__(self, n_inputs, nrons, n_outputs):
		self.n_inputs = n_inputs
		self.nrons = nrons
		self.n_outputs = n_outputs
		self.gen_w8s()
		self.gen_bias()

	def __str__(self):
		return str(self.__dict__)

	def gen_w8s(self):
		self.w1	= 0.1*np.random.randn(self.n_inputs,self.nrons)
		self.w2	= 0.1*np.random.randn(self.n_outputs,self.nrons)
	def gen_bias(self):
		self.b1	= 0.1*np.random.randn(self.n_outputs,self.nrons)
		self.b2	= 0.1*np.random.randn()

	def sigmoid(self,x):
		return 1.0/(1+ np.exp(-x))

	def think(self, X):			# X = [i1, i2, i3, i4, i5]
		X = np.array(X)
		X_norm = (X-X.mean())/X.std()	# This shit rocks
		# print(X_norm)
		z = (np.dot(X_norm,self.w1)+self.b1)
		# z = np.tanh(z)
		a = np.dot(z,self.w2.T)+self.b2
		return self.sigmoid(a)
		# return max(0,a)