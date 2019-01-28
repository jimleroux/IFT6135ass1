# -*- coding: utf-8 -*-

"""
MLP
"""
__authors__ = "Jimmy Leroux"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux"
__studentid__ = "1024610"

import torch
if torch.cuda.is_available():
    import cupy as np
else:
    import numpy as np


class NN(object):

    def __init__(
            self, hidden_dims=(1024, 2048), n_hidden=2, mode='train',
            datapath=None, model_path=None):

        self.n_hidden = n_hidden
        self.hidden_dims = hidden_dims
        self.parameters = {}
        self.layers = [28*28, 512, 512, 10]
        self.initialize_weights(n_hidden, "normal" )

    def initialize_weights(self, n_hidden, mode="glorot"):
        num_layer = len(self.layers)
        if mode == "zero":
            for i in range(1, num_layer):
                self.parameters["W"+str(i)] = np.zeros(
                    (self.layers[i], self.layers[i-1]))
                self.parameters["b"+str(i)] = np.zeros((self.layers[i], 1))
        if mode == "normal":
            for i in range(1, num_layer):
                self.parameters["W"+str(i)] = np.random.rand(
                    self.layers[i], self.layers[i-1])
                self.parameters["b"+str(i)] = np.zeros((self.layers[i], 1))
        if mode == "glorot":
            for i in range(1, num_layer):
                dl = 1. / np.sqrt(6./(self.layers[i-1]+self.layers[i]))
                self.parameters["W"+str(i)] = np.ones(
                    (self.layers[i], self.layers[i-1])) * np.random.uniform(
                    -dl, dl, (self.layers[i], self.layers[i-1]))
                self.parameters["b"+str(i)] = np.zeros((self.layers[i], 1))

    def forward(self, X):
        """
        Forward propagation method. It propagated X through the network.

        Parameters:
        -----------
        X: Input matrix we wish to propagate. Shape: (dim, num_exemple)

        Returns:
        cache: Dictionary of the intermediate cache at each step
                of the propagation.
        """

        h1 = np.dot(self.parameters["W1"], X) + self.parameters["b1"]
        a1 = self.activation(h1)
        h2 = np.dot(self.parameters["W2"], a1) + self.parameters["b2"]
        a2 = self.activation(h2)
        h3 = np.dot(self.parameters["W3"], a2) + self.parameters["b3"]
        a3 = self.softmax(h3)
        cache = {
                "h1": h1, "a1": a1, "h2": h2, "a2": a2,
                "h3": h3, "a3": a3, "X": X
                }
        return cache

    def activation(self, X):
        return np.maximum(0, X)

    def loss(self, Y, cache, lam):
        loss = np.sum(-np.log(cache["a3"])*Y)
        loss += lam*np.sum(np.abs(self.parameters["W1"]))\
            + lam*np.sum(np.abs(self.parameters["W2"]))\
            + lam*np.sum(np.abs(self.parameters["W3"]))\
            + lam*np.sum(self.parameters["W1"]**2)\
            + lam*np.sum(self.parameters["W2"]**2)\
            + lam*np.sum(self.parameters["W3"]**2)
        loss *= cache["X"].shape[1]
        return loss

    def softmax(self, X):
        max_ = X.max(axis=0)
        out = np.exp(X-max_) / np.sum(np.exp(X-max_), axis=0)
        return out

    def backward(self, cache, Y, lam):
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

        dh3 = cache["a3"] - Y
        dW3 = np.mean(dh3[:, None, :]*cache["a2"][None, :, :], axis=2)\
            + 2*lam*self.parameters["W3"]\
            + lam*np.sign(self.parameters["W3"])
        db3 = np.mean(dh3, axis=1, keepdims=True)

        da2 = np.sum(dh3[:, None, :]*self.parameters["W3"][:, :, None], axis=0)
        dh2 = da2*1.*(cache["a2"] > 0)
        dW2 = np.mean(dh2[:, None, :]*cache["a1"][None, :, :], axis=2)\
            + 2*lam*self.parameters["W2"]\
            + lam*np.sign(self.parameters["W2"])
        db2 = np.mean(dh2, axis=1, keepdims=True)

        da1 = np.sum(dh2[:, None, :]*self.parameters["W2"][:, :, None], axis=0)
        dh1 = da1*1.*(cache["a1"] > 0)
        dW1 = np.mean(dh1[:, None, :]*cache["X"][None, :, :], axis=2)\
            + 2*lam*self.parameters["W1"]\
            + lam*np.sign(self.parameters["W1"])
        db1 = np.mean(dh1, axis=1, keepdims=True)

        dx = np.sum(dh1[:, None, :]*self.parameters["W1"][:, :, None], axis=0)
        grads = {
            "dW3": dW3, "db3": db3, "dW2": dW2, "dW1": dW1,
            "db2": db2, "db1": db1, "dx": dx
            }
        return grads

    def update(self, grads, lr):
        for par in self.parameters.keys():
            self.parameters[par] -= lr * grads["d"+par]

    def train(self, trainloader, testloader,
              num_epoch=10, lr=0.0001, lam=0.0001):
        acc_train = []
        acc_test = []
        for epoch in range(num_epoch):
            print("Epoch:{}".format(epoch))
            for data in trainloader:
                inputs, labels = data
                inputs = self.transform_input(inputs)
                labels = self.onehot(labels)
                cache = self.forward(inputs)
                grads = self.backward(cache, labels, lam)
                self.update(grads, lr)

            # Calculate the accuracy
            correct = 0.
            total = 0.
            for data in trainloader:
                inputs, labels = data
                inputs = self.transform_input(inputs)
                labels = self.onehot(labels)
                cache = self.forward(inputs)
                preds = np.argmax(cache["a3"], axis=0)
                correct += np.sum(preds == np.argmax(labels, axis=0))
                total += labels.shape[1]
                if total > 5000:
                    break
            acc_train.append(float(correct/total)*100)

            correct = 0.
            total = 0.
            for data in testloader:
                inputs, labels = data
                inputs = self.transform_input(inputs)
                labels = self.onehot(labels)
                cache = self.forward(inputs)
                preds = np.argmax(cache["a3"], axis=0)
                correct += np.sum(preds == np.argmax(labels, axis=0))
                total += labels.shape[1]
                if total > 5000:
                    break
            acc_test.append(float(correct/total)*100)
            print("Accuracy train: {0:.2f}, Accuracy valid: {1:.2f}".format(
                acc_train[epoch], acc_test[epoch]))
        return acc_train, acc_test

    def test(self):
        pass

    def onehot(self, Y):
        onehot = np.zeros((self.layers[-1], len(Y)))
        for j in range(len(Y)):
            onehot[int(Y[j]), j] = 1.
        return onehot

    def transform_input(self, X):
        return np.array(X.reshape((X.shape[0], self.layers[0])).numpy().T)
