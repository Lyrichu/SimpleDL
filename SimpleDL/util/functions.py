#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: functions.py
@time: 2018/09/13 12:41
@description:
useful functions collections
"""
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (1-np.exp(-x))/(1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)


def softmax(x):
    y = np.exp(x)
    return y/np.sum(y,axis=0)









