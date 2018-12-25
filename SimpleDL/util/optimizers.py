#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: optimizers.py
@time: 2018/09/13 13:33
@description:
various optimizers
"""
import numpy as np
from SimpleDL.util.config import *
from SimpleDL.util.core import *

def _gradient(x, y, target_func, loss_func):
    grads = np.zeros_like(global_variables_list)
    variables_num = len(global_variables_list)
    for i in range(variables_num):
        variables_list1 = np.copy(global_variables_list)
        variables_list2 = np.copy(global_variables_list)
        variables_list1[i] += delta
        variables_list2[i] -= delta
        y1 = target_func(x, *transform(variables_list1))
        y2 = target_func(x, *transform(variables_list2))
        grad = (loss_func(y1, y) - loss_func(y2, y)) / (2 * delta)
        grads[i] = grad

    return grads


def gradient_descent(x, y, target_func, loss_func, lr=lr):
    global global_variables_list
    grads = _gradient(x, y, target_func, loss_func)
    global_variables_list -= lr * grads
    return transform(global_variables_list)



def momentum(x,y,target_func,loss_func,lr):
    global global_variables_list,global_momemtum_grads
    grads = _gradient(x, y, target_func, loss_func)
    if len(global_momemtum_grads) == 0:
        global_momemtum_grads = np.zeros_like(grads)
    else:
        global_momemtum_grads = momentum_beta*global_momemtum_grads + lr*grads
    global_variables_list -=global_momemtum_grads
    return transform(global_variables_list)


def adagrad(x,y,target_func,loss_func,lr):
    global global_variables_list,global_adagrad_s
    grads = _gradient(x, y, target_func, loss_func)
    if len(global_adagrad_s) == 0:
        global_adagrad_s = np.square(grads)
    else:
        global_adagrad_s += np.square(grads)
    global_variables_list -= lr*grads/np.sqrt(global_adagrad_s+adagrad_eps)
    return transform(global_variables_list)



def rmsprop(x,y,target_func,loss_func,lr):
    global global_rmsprop_s,global_variables_list
    grads = _gradient(x, y, target_func, loss_func)
    if len(global_rmsprop_s) == 0:
        global_rmsprop_s = (1-rmsprop_beta)*np.square(grads)
    else:
        global_rmsprop_s = rmsprop_beta*global_rmsprop_s + (1-rmsprop_beta)*np.square(grads)

    global_variables_list -= lr * grads/np.sqrt(global_rmsprop_s + rmsprop_eps)
    return transform(global_variables_list)


def adam(x,y,target_func,loss_func,lr,epoch):
    global global_adam_m,global_adam_s,global_variables_list
    grads = _gradient(x, y, target_func, loss_func)
    if len(global_adam_s) == 0 or len(global_adam_m) == 0:
        global_adam_m = (1-adam_beta1)*grads
        global_adam_s = (1-adam_beta2)*np.square(grads)
    else:
        global_adam_m = adam_beta1*global_adam_m + (1-adam_beta1)*grads
        global_adam_s = adam_beta2*global_adam_s + (1 - adam_beta2) * np.square(grads)
    global_adam_m /= (1-adam_beta1**(epoch+1))
    global_adam_s /= (1-adam_beta2**(epoch+1))
    global_variables_list -= lr*global_adam_m/np.sqrt(global_adam_s+adam_eps)
    return transform(global_variables_list)


