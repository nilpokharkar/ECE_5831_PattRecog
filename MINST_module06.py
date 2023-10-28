#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:25:07 2022

@author: Nilakshi Pokharkar
"""
import urllib.request
import gzip
import numpy as np
#import matplotlib.pyplot as plt
import os
import pickle

class MNIST:
    img_size = 784 #28*28
    img_Dim = (1, 28 ,28)
    minst_pkl_filename = 'mnist_dataset.pkl'
    key_file = {
                'train_img':'train-images-idx3-ubyte.gz',
                'train_label':'train-labels-idx1-ubyte.gz',
                'test_img': 't10k-images-idx3-ubyte.gz',
                'test_label': 't10k-labels-idx1-ubyte.gz'
            }
    
    def __init__(self):
        self.network = None
        
    # Download the datasets from the URL in the current working repository
    def download_dataset(self):
        url_base = 'http://yann.lecun.com/exdb/mnist/'
        
        for value in self.key_file.values():
            if os.path.exists(value):
                print('File exists!')
            else:
                print('Downloading {}.....'.format(value))
                urllib.request.urlretrieve(url_base + value, value)
                print('Download complete!!!')
                
    def load_images(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16) #This data can be found in the dataset URL
        images = images.reshape(-1, self.img_size)
        
        print('Done with image loading: ', file_name)
        return images        
    
    def load_labels(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        print('Done with label load: ', file_name)
        return labels
    
    #Creates an array of 10 itmes making all 0s excpet that true label 
    #i.e if true label is 2 then hot_label will be [0,0,1,0,0,0,0,0,0,0]
    def change_one_hot_label(self, x):
        t = np.zeros((x.size, 10))
        for idx, row in enumerate(t):
            row[x[idx]] = 1
            
        return t
    
    def init_mnist(self):
        self.download_dataset()
        datasets = {}
        datasets['train_images'] = self.load_images(self.key_file['train_img'])
        datasets['train_labels'] = self.load_labels(self.key_file['train_label'])
        datasets['test_images'] = self.load_images(self.key_file['test_img'])
        datasets['test_labels'] = self.load_labels(self.key_file['test_label'])
        
        #Download datasets and use it again so that no need of redownloading and unzipping.
        print('Creating a pickle for the data....')
        with open(self.minst_pkl_filename,'wb') as f:
            pickle.dump(datasets, f, -1)
        print('Pickle for dataset created!!')
        
    def load_mnist(self, normalize=False, flatten=True, one_hot_label=False):
        if not os.path.exists(self.minst_pkl_filename):
            self.init_mnist()
            
        with open(self.minst_pkl_filename,'rb') as f:
            datasets = pickle.load(f)
        
        #Originally the value swill be b/w 0-255. Normalized by '/255.0' will make the value b/w 0-1(black-white).
        if normalize:
            for key in ('train_images', 'test_images'):
                datasets[key] = datasets[key].astype(np.float32)
                datasets[key] /= 255.0
                
        if one_hot_label:
            datasets['train_labels'] = self.change_one_hot_label(datasets['train_labels'])
            datasets['test_labels'] = self.change_one_hot_label(datasets['test_labels'])
        
        #28*28 changed to 1D(784)    
        if not flatten:
            for key in ('train_images', 'test_images'):
                datasets[key] = datasets[key].reshape(-1, -1, 28, 28)
                
        return((datasets['train_images'], datasets['train_labels']), (datasets['test_images'], datasets['test_labels']))
    
if __name__ == '__main__':
    mnist = MNIST()
    mnist.init_mnist()