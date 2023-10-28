#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:34:39 2022

@author: Nilakshi Pokharkar
"""
import numpy as np 
from tqdm import tqdm
import pickle

import activations
import loss_functions
import gradients


#weight_init_std-> standard deviation
#hidden_size-> number of nodes in hidden layer
#input_size-> considering MNIST data set. image dimensions are 28*28
#output_size-> considering MNIST data set. output labels can be from 0-9 thus 10 classes
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params={}
        self.params['w1']=weight_init_std\
                          *np.random.randn(input_size, hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['w2']=weight_init_std\
                          *np.random.randn(hidden_size, output_size)
        self.params['b2']=np.zeros(output_size)
        
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
    
    #Calculating forward propogation    
    def prediction(self, x):
        w1, w2 = self.params['w1'],self.params['w2']
        b1, b2 = self.params['b1'],self.params['b2']
        
        a1 = np.dot(x, w1) + b1
        z1 = activations.sigmoid(a1)
        
        a2 = np.dot(z1, w2) + b2
        z2 = activations.softmax(a2)
        
        return z2
        
    def loss(self, x, t):
        z = self.prediction(x)
        
        loss = loss_functions.cross_entropy_error(z,t)
        return loss
    
    #Calculates the numerical gradient/slopes for w1, b1, w2, b2
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        
        grads={} #dictionary
        grads['w1'] = gradients.numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = gradients.numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = gradients.numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = gradients.numerical_gradient(loss_w, self.params['b2'])
        
        return grads
    
    #Backpropogation
    def gradient(self, x, t):
        w1, w2 = self.params['w1'],self.params['w2']
        b1, b2 = self.params['b1'],self.params['b2']
        
        grads={}
        
        batch_num = x.shape[0]
        
        #forward
        a1 = np.dot(x, w1) + b1
        z1 = activations.sigmoid(a1)
        
        a2 = np.dot(z1, w2) + b2
        z2 = activations.softmax(a2)
        
        #backward
        dz2 = (z2-t)/batch_num
        grads['w2'] = np.dot(z1.T, dz2)
        grads['b2'] = np.sum(dz2, axis=0)
        
        dz1 = np.dot(dz2, w2.T)
        da1 = activations.sigmoid_grad(a1)*dz1
        grads['w1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        
        return grads
    
    def accuracy(self, x, t):
        y = self.prediction(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        return np.sum(y == t)/float(x.shape[0])
    
    #fit means train
    def fit(self, iterations, x_train, t_train, x_test, t_test, batch_size, learning_rate=0.1, back_propogation=True):
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size/batch_size, 1)
        
        print("Start Training....")
        for i in tqdm(range(iterations)):
            #get mini batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            if back_propogation == True:
                #calculate gradient
                grad = self.gradient(x_batch, t_batch)
            else:
                grad = self.numerical_gradient(x_batch, t_batch)
            
            for key in ('w1','b1','w2','b2'):
                self.params[key]  -= learning_rate*grad[key]
                
            loss = self.loss(x_batch, t_batch)
            self.train_losses.append(loss)
            
            #After every mini batch
            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                self.train_accs.append(train_acc)
                self.test_accs.append(test_acc)
            
        print("Training Completed!!")
        
    def save_model(self, model_filename):
        with open(model_filename, 'wb') as f:
            pickle.dump(self.params, f, -1)
        print('Model Saved!!')
        
    def load_model(self, model_filename):
        with open(model_filename,'rb') as f:
            self.params = pickle.load(f)
            
        #print('Model Loaded!!')
#%%    
# net = TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)
# # print('w1={}'.format(net.params['w1'].shape)) #//output: w1=(784, 50)
# # print('b1={}'.format(net.params['b1'].shape)) #//output: b1=(50,)
# # print('w2={}'.format(net.params['w2'].shape)) #//output: w2=(50, 10)
# # print('b2={}'.format(net.params['b2'].shape)) #//output: b2=(10,)

# #SInce we will be using MNIST dataset this x will be the image input and not any random matix as shown. Below lines are just for testing.
# x = np.random.randn(100, 784) #Return 100 of 784 random values. 
# y = net.prediction(x)

# t = np.random.randn(100, 10)
# grads = net.numerical_gradient(x, t)
