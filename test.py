#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 01:49:44 2022

@author: Nilakshi Pokharkar
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
#%%

from two_layer_net import TwoLayerNet
#from MINST_module06 import MNIST 

trained_model_filename = 'npokhark_mnist_nn_model.pkl'

#LOAD MNIST TEST DATASETS
# mnist = MNIST()
# (x_train, t_train),(x_test, t_test) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=True)


#%%

network = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

network.load_model(trained_model_filename)
#print(network.params)
# output = network.prediction(x_test[2987])
# print(np.argmax(output))

#%%

file_name = sys.argv[1]
digit_Value = None

image = cv2.imread('../TestImages/{}'.format(file_name))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)
image = cv2.resize(image,(28,28))
#print(image.shape)

plt.imshow(image, cmap ='gray') 

#plt.figure() #display the image in jupyter notebook
#plt.show() #displays the figure size info
#plt.axis('off')
image = image.reshape(784,)
    
output = network.prediction(image)
predicted_num = np.argmax(output)

if len(sys.argv) == 3:
    digit_Value = int(sys.argv[2])
    
    if(predicted_num == digit_Value):
        print('Image {} is for digit {} is recognized as {}'.format(file_name, digit_Value, predicted_num))
    else:
        print('Image {} is for digit {} but the inference result is {}'.format(file_name, digit_Value, predicted_num))
# else:
#     print('{}={}'.format(file_name,predicted_num))
    