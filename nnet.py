#!/usr/bin/env python3
import numpy as np

sd=394#np.random.randint(500)	# 394
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
		self.w1	= 0.1*np.random.randn(self.n_inputs,self.nrons).astype(np.float32)		# (8,30)
		self.w2	= 0.1*np.random.randn(self.nrons,self.n_outputs).astype(np.float32)		# (30,1)
	def gen_bias(self):
		self.b1	= 0.1*np.random.randn(1,self.nrons).astype(np.float32)
		self.b2	= 0.1*np.random.randn(1,1).astype(np.float32)

	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))

	def think(self, X):
		X = np.array(X,dtype=np.float32)					# (1,8)
		self.X_norm = (X-X.mean())/X.std()	# This shit rocks
		# print(self.X_norm.shape)
		self.z = (np.dot(self.X_norm,self.w1)+self.b1)	# (1,30)
		# self.z = self.sigmoid(self.z)
		a = np.dot(self.z,self.w2)+self.b2
		self.out=self.sigmoid(a)
		return self.out
		# return max(0,a)