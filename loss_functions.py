#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:55:18 2022

@author: Nilakshi Pokharkar
"""
import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        
    if(t.size == y.size):
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta))/batch_size