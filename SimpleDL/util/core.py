#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: core.py
@time: 2018/09/13 13:00
@description:
core functions
"""
import numpy as np
from SimpleDL.util.config import global_variables_list
from SimpleDL.util.config import global_variables_shape
from SimpleDL.util.optimizers import *


def transform(variables_list = global_variables_list):
    vars = []
    index = 0
    if isinstance(variables_list,list):
        variables_list = np.array(variables_list)
    for shape in global_variables_shape:
        size = 1
        for i in shape:
            size *= i
        vars.append(variables_list[index:index+size].reshape(shape))
        index += size
    return vars


def onehot_encoder(x,num_classes):
    return np.eye(num_classes)[x].T



def conv2d(x,w,kernel_size,input_dim,output_dim,step_size,padding = 'same'):
    assert len(x.shape) == 4,"input data should have 4 dimensions!"
    assert len(w.shape) == 4,"conv kernel should have 4 dimensions!"
    assert x.shape[-1] == input_dim,"input_dim should be %d!" % x.shape[-1]
    assert w.shape[0] == output_dim,"output dim should be %d!" % w.shape[-1]
    assert kernel_size == w.shape[1:3],"kernel size should be %s" %(w.shape[1:3])
    assert padding == 'same' or padding == 'valid',"padding should be 'same' or 'valid'!"
    x_num,x_w,x_h,x_c = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
    w_o,w_w,w_h,w_i = w.shape[0],w.shape[1],w.shape[2],w.shape[3]
    if padding == 'valid':
        conv_w = (x_w-w_w)//step_size+1
        conv_h = (x_h-w_h)//step_size+1
    else:
        w_padding_size = (x_w-1)*step_size+w_w-x_w
        h_padding_size = (x_h-1)*step_size+w_h-x_h
        conv_w = x_w
        conv_h = x_h
        x1 = np.copy(x)
        x = np.zeros((x_num,x_w+w_padding_size,x_h+h_padding_size,x_c))
        x[:,w_padding_size//2:w_padding_size//2+x_w,h_padding_size//2:h_padding_size//2+x_h,:] = x1
    conv_matrix = np.zeros((x_num, conv_w, conv_h, w_o))
    for i in range(x_num):
        for j in range(w_o):
            for k in range(conv_w):
                for l in range(conv_h):
                    conv_matrix[i,k,l,j] = (w[j,:,:,:] * x[i,k*step_size:k*step_size+w_w,k*step_size:k*step_size+w_h,:]).sum()


    return conv_matrix








