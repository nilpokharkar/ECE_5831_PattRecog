#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 00:54:31 2022

@author: Nilakshi Pokharkar
"""
#Below module has a definition of Two Layer Neural Network
from two_layer_net import TwoLayerNet
#Below module has a definition of MNIST class 
from MINST_module06 import MNIST 

import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm #progress bar

trained_model_filename = 'npokhark_mnist_nn_model.pkl'

#LOAD MNIST DATASETS
mnist = MNIST()
(x_train, t_train),(x_test, t_test) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=True)

#hyperparameters
iterations = 20000
batch_size = 100
learning_rate = 0.05

hidden_size=50
network = TwoLayerNet(input_size=mnist.img_size, hidden_size=hidden_size, output_size=10)

#training the network
network.fit(iterations, x_train, t_train, x_test, t_test, batch_size, learning_rate, back_propogation=True)

network.save_model(trained_model_filename)

plt.figure()
x = np.arange(len(network.train_losses))
plt.plot(x, network.train_losses, label='train loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.figure()
plt.show()

plt.figure()
marker = {'train':'o','test':'s'}
x = np.arange(len(network.train_accs))
plt.plot(x, network.train_accs, label='train acc')
plt.plot(x, network.test_accs, label='test loss', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.figure()
plt.show()
