#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: data.py
@time: 2018/09/13 12:02
@description:
generate example fake datas
"""
import numpy as np

MNIST_DATA_PATH = "/".join(__file__.split("/")[:-2]) + "/resources/mnist"

def _func_sin_cos_exp(x):
    return np.sin(x[0]) + np.cos(x[1]) + np.exp(x[0]+x[1]) + 1


def generate_data_sin_cos_exp(N = 100):
    data = np.random.random((2,N))
    labels = np.zeros((1,N))
    for i in range(N):
        labels[0,i] = _func_sin_cos_exp(data[:,i])
    data = np.concatenate((data,labels),0)
    return data



def generate_simple_binary_data(N = 100):
    data = np.random.random((2,N))
    labels = np.zeros((1,N))
    for i in range(N):
        label = 0 if np.sin(data[0,i]) + data[1,i] < 1 else 1
        labels[0,i] = label

    data = np.concatenate((data,labels),0)
    return data


def load_mnist(path=MNIST_DATA_PATH):
    x_train = np.load(MNIST_DATA_PATH+"/x_train.npy")
    x_test = np.load(MNIST_DATA_PATH+"/x_test.npy")
    y_train = np.load(MNIST_DATA_PATH+"/y_train.npy")
    y_test = np.load(MNIST_DATA_PATH+"/y_test.npy")
    return x_train,x_test,y_train,y_test







