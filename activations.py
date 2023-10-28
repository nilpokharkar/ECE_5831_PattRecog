#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:46:52 2022

@author: Nilakshi Pokharkar
"""
import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-a))

#This will only work with 1D input. WIth this function loss value will not decrease. 
def softmax_1d(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#sigmoid for backpropogation
def sigmoid_grad(a):
    return (1.0 - sigmoid(a)) * sigmoid(a)