# -*- coding: utf-8 -*-

"""
MLP
"""
__authors__ = "Jimmy Leroux"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux"
__studentid__ = "1024610"

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import csv
import os



class NN(object):
    
    def __init__(
    		self, hidden_dims=(1024,2048), n_hidden=2, mode='train',
    		datapath=None, model_path=None):
    	
    	self.n_hidden = n_hidden
    	self.hidden_dims = hidden_dims
    	self.parameters = {}
    	self.layers = [28*28, 512, 512, 10]

	def initialize_weights(self, n_hidden, dims):
		num_layer = len(self.layers)
		for i in range(1, num_layer):
			n_c = 1. / np.sqrt(self.layers[i])
			self.parameters["W"+str(i)] = np.ones(
				(self.layers[i], self.layers[i-1])) * np.random.uniform(
				-n_c, n_c, (self.layers[i], self.layers[i-1]))
			self.parameters["b"+str(i)] = np.zeros((self.layers[i], 1))

	def forward(self, X):
		"""
		Forward propagation method. It propagated X through the network.

		Parameters:
		-----------
		X: Input matrix we wish to propagate. Shape: (dim, num_exemple)

		Returns:
		cache: Dictionary of the intermediate cache at each step of the propagation.
		"""

		h1 = np.dot(self.parameters["W1"], X) + self.parameters["b1"]
		a1 = self.activation(h1)
		h2 = np.dot(self.parameters["W2"], a1) + self.parameters["b2"]
		a2 = self.activation(h2)
		h3 = np.dot(self.parameters["W3"], a2) + self.parameters["b3"]
		a3 = self.softmax(h3)
		cache = {"h1":h1, "a1":a1, "h2":h2, "a2":a2, "h3":h3, "a3":a3, "X":X}
		return cache

	def activation(self, X):
		return np.maximum(0, X)

	def loss(self, prediction):
		pass

	def softmax(self, X):
		max_ = X.max(axis=0)
		out = np.exp(X - max_) / np.sum(np.exp(X - max_), axis=0)
		return out

	def backward(self, cache, Y):
		"""
		Method performing the backpropagation for the model.
		
		Parameters:
		-----------
		cache: Stored intermediate cache of the forward propagation pass.
		Y: Target cache (of the training set). Shape: (num_class, num_exemple)

		Returns:
		--------
		grads: Dictionary containing the gradients of the parameters.

		"""

		grads = {}
		dh3 = cache["h3"] - Y
		dW3 = np.mean(dh3[:,None,:] * cache["h2"][None,:,:], axis=2) +\
			2 * self.lams[3] * self.parameters["W3"] +\
			self.lams[2] * np.sign(self.parameters["W3"])
		db3 = np.mean(dh3, axis=1, keepdims=True)
		
		da2 = np.sum(dh3[:,None,:] * self.parameters["W2"][:,:,None], axis=0)
		dh2 = da2 * 1. * (cache["a2"]>0)
		dW2 = np.mean(dh_a[:,None,:] * cache["X"][None,:,], axis=2) +\
			2 * self.lams[1] * self.parameters["W1"] +\
			self.lams[0] * np.sign(self.parameters["W1"])
		db2 = np.mean(dh_a, axis=1, keepdims=True)

		da1 = np.sum(dh2[:,None,:] * self.parameters["W1"][:,:,None], axis=0)
		dh1 = da1 * 1. * (cache["a1"]>0)
		dW1 = np.mean(dh_a[:,None,:] * cache["X"][None,:,], axis=2) +\
			2 * self.lams[1] * self.parameters["W1"] +\
			self.lams[0] * np.sign(self.parameters["W1"])
		db2 = np.mean(dh_a, axis=1, keepdims=True)
		
		dx = np.sum(dh1[:,None,:]*self.parameters["W1"][:,:,None], axis=0)
		grads = {"dW2":dW2, "dW1":dW1, "db2":db2, "db1":db1, "dx":dx}
		return grads

	def update(self, grads):
		pass

	def train(self):
		pass

	def test(self):
		pass