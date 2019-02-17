# -*- coding: utf-8 -*-
import torch

if torch.cuda.is_available():
    import cupy as np
else:
    import numpy as np


class NeuralNetwork(object):
    """
    Implementation of a custom mlp.
    """

    def __init__(
            self, hidden_dims=(1024, 2048), n_hidden=2, mode="glorot"):
        """
        Parameters:
        -----------
        hidden_dims (tuple): Dimension of the hidden layers.
        n_hidden (int): Number of hidden layers.
        """

        self.n_hidden = n_hidden
        self.hidden_dims = hidden_dims
        self.parameters = {}
        self.layers = [28*28, 512, 512, 10]
        self.init = mode
        self.initialize_weights(mode=self.init)

    def initialize_weights(self, mode="glorot"):
        """
        Weights initialization method. Used at the begining of the training
        step

        Parameters:
        -----------
        mode (str): Initilization method.
        """

        num_layer = len(self.layers)
        if mode == "zero":
            for i in range(1, num_layer):
                self.parameters["W"+str(i)] = np.zeros(
                    (self.layers[i], self.layers[i-1]))
                self.parameters["b"+str(i)] = np.zeros((self.layers[i], 1))
        if mode == "normal":
            for i in range(1, num_layer):
                self.parameters["W"+str(i)] = np.random.normal(
                    size=(self.layers[i], self.layers[i-1]))
                self.parameters["b"+str(i)] = np.zeros((self.layers[i], 1))
        if mode == "glorot":
            for i in range(1, num_layer):
                dl = np.sqrt(6./(self.layers[i-1]+self.layers[i]))
                self.parameters["W"+str(i)] = np.ones(
                    (self.layers[i], self.layers[i-1])) * np.random.uniform(
                    -dl, dl, (self.layers[i], self.layers[i-1]))
                self.parameters["b"+str(i)] = np.zeros((self.layers[i], 1))

    def forward(self, X):
        """
        Forward propagation method.

        Parameters:
        -----------
        X (array): Input array we wish to propagate. Shape: (dim, num_exemple)

        Returns:
        --------
        cache (dict): Dictionary of the intermediate cache at each step
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
        """
        Activation function, here its ReLU.

        Parameters:
        -----------
        X (array): Array contraining data. Shape: (dim, num_exemple)

        Returns:
        --------
        out (array): Array of the activated data.
        """

        out = np.maximum(0, X)
        return out

    def loss(self, Y, cache, lam):
        """
        Calculate the loss on our model.

        Parameters:
        -----------
        Y (array): Labels/target of the forwarded data.
                    Shape: (dim, num_exemple)
        cache (dict): Dictionary containing the intermediate values.
        lam (float): Regularisation constant. To be more rigorous, we
                        should add a lam for the L1 and L2 regularisation.

        Returns:
        --------
        loss (float): Value of the total loss.
        """

        loss = np.sum(-np.log(cache["a3"])*Y)
        loss += lam*np.sum(np.abs(self.parameters["W1"]))\
            + lam*np.sum(np.abs(self.parameters["W2"]))\
            + lam*np.sum(np.abs(self.parameters["W3"]))\
            + lam*np.sum(self.parameters["W1"]**2)\
            + lam*np.sum(self.parameters["W2"]**2)\
            + lam*np.sum(self.parameters["W3"]**2)
        # loss *= cache["X"].shape[1]
        return loss

    def softmax(self, X):
        """
        Softmax activation. Used in the last layer of the NN for the
        classification.

        Parameters:
        -----------
        X (array): Array containing the data. Shape: (dim, num_exemple)

        Returns:
        --------
        out (array): "probability" of each class for each data exemple.
                        Shape: (dim, num_exemple).
        """

        # We use a little trick for stability purpose.
        max_ = X.max(axis=0)
        out = np.exp(X-max_) / np.sum(np.exp(X-max_), axis=0)
        return out

    def backward(self, cache, Y, lam):
        """
        Method performing the backpropagation for the model.

        Parameters:
        -----------
        cache (dict): Stored intermediate cache of the forward
                        propagation pass.

        Y (array): Targets of the probagated exemple.
                    Shape: (num_class, num_exemple)
        lam (float): Regularization constant.

        Returns:
        --------
        grads (dict): Dictionary containing the gradients of the parameters.

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

    def grad_check(self, X, Y, epsilon):
        cache = self.forward(X)
        grad = self.backward(cache, Y, 0)
        gradtest = np.zeros(grad["dW3"].shape)

        for i in range(gradtest.shape[0]):
            for j in range(10):
                #  Add to the parameters epsilon
                self.parameters["W3"][i, j] += epsilon
                cache = self.forward(X)
                #  Calculate the loss
                loss1 = self.loss(Y, cache, 0)
                #  Move back the parameters
                self.parameters["W3"][i, j] -= 2 * epsilon
                cache = self.forward(X)
                #  Recalculate the loss
                loss2 = self.loss(Y, cache, 0)
                #  We divide by X.shape[1] in case we want to check the
                #  gradient with minibatches.
                gradtest[i, j] = (loss1-loss2) / (2*epsilon) / X.shape[1]
                #  We reset the parameters where they were.
                self.parameters["W3"][i, j] += epsilon
        return gradtest[:, :10], grad["dW3"][:, :10]

    def update(self, grads, lr):
        """
        Update method for the model.

        Parameters:
        -----------
        grads (dict): Dictionary containing the gradients of each
                        parameters.
        lr (float): Learning rate for the gradient descent.
        """

        for par in self.parameters.keys():
            self.parameters[par] -= lr*grads["d"+par]

    def train(
            self, train, test, num_epoch=10,
            lr=0.1, lam=0.0000, batchsize=256):
        """
        Training method.

        Parameters:
        -----------
        train (torch dataset): Torch dataset of the training data.
        test (torch dataset): Torch dataset of the test data.
        num_epoch (int): Number of epoch in the training phase.
        lr (float): Learning rate.
        lam (float): Regularisation constant.
        batchsize (int): Number of example in a minibatch.

        Returns:
        --------
        acc_train (list): List containing the accuracy of the network
                            at each epoch.
        acc_test (list): List containing the accuracy of the network
                            at each epoch.
        """

        trainloader = torch.utils.data.DataLoader(
            train, batch_size=batchsize, shuffle=True)
        testloader = torch.utils.data.DataLoader(
            test, batch_size=batchsize, shuffle=False)

        err_train = []
        err_test = []
        loss_train = []
        loss_test = []

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
            losses = 0.
            for data in trainloader:
                inputs, labels = data
                inputs = self.transform_input(inputs)
                labels = self.onehot(labels)
                cache = self.forward(inputs)
                preds = np.argmax(cache["a3"], axis=0)
                correct += np.sum(preds == np.argmax(labels, axis=0))
                total += labels.shape[1]
                losses += self.loss(labels, cache, lam)
            err_train.append(1 - float(correct/total))
            loss_train.append(float(losses) / total)
            
            correct = 0.
            total = 0.
            losses = 0.
            for data in testloader:
                inputs, labels = data
                inputs = self.transform_input(inputs)
                labels = self.onehot(labels)
                cache = self.forward(inputs)
                preds = np.argmax(cache["a3"], axis=0)
                correct += np.sum(preds == np.argmax(labels, axis=0))
                total += labels.shape[1]
                losses += self.loss(labels, cache, lam)
            err_test.append(1 - float(correct/total))
            loss_test.append(float(losses) / total)
            print("Error train: {0:.2f}, Error valid: {1:.2f}".format(
                loss_train[epoch], loss_test[epoch]))
        return err_train, err_test, loss_train, loss_test

    def test(self):
        pass

    def onehot(self, Y):
        """
        Usefull function to map labels to onehot vectors.

        Parameters:
        -----------
        Y (array): Array containing the labels to put in onehot form.
                    Shape: (1, num_exemple)

        Returns:
        --------
        onehot (array): Onehot encoding of all the labels Y.
                        Shape: (num_classes, num_exemple)
        """

        onehot = np.zeros((self.layers[-1], len(Y)))
        for j in range(len(Y)):
            onehot[int(Y[j]), j] = 1.
        return onehot

    def transform_input(self, X):
        """
        Method to transform the data from torch tensor to numpy.

        Parameters:
        -----------
        X (array): Array containing the data of shape: (num_exemple, dim)

        Returns:
        --------
        transformed (array): Array containing the tranformed data.
                                Shape: (dim, num_exemple)
        """

        transformed = np.array(
            X.reshape((X.shape[0], self.layers[0])).numpy().T)
        return transformed
