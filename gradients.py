#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:18:43 2022

@author: Nilakshi Pokharkar
"""

import numpy as np

#supports multi dimensional
def numerical_gradient(func, x): #assuming x is a vector i.e multiple values
    h = 1e-4 #0.00001
    grad = np.zeros_like(x)
    
    #new iteration method for multidimension
    it = np.nditer(x, flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
    #for idx in range(x.size):
        idx = it.multi_index
        temp = x[idx]
        
        #for f(x+h)
        x[idx] = temp + h
        fxh1 = func(x)
        
        #for f(x-h)
        x[idx] = temp -h
        fxh2 = func(x)
        
        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = temp
        
        it.iternext()
        
    return grad