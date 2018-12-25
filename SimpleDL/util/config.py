#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: config.py
@time: 2018/09/13 11:45
@description:
global configuration
"""
import numpy as np

global_variables_list = []
global_variables_shape = []
global_momemtum_grads = []
global_adagrad_s = []
global_rmsprop_s = []
global_adam_m = []
global_adam_s = []
init_variables_uniform = np.random.uniform
init_variables_random = np.random.random

lr = 0.01
lr_schedule = 0.5
lr_schedule_epochs = 100
lr_init = 1.
epochs = 500
delta = 1e-8
momentum_beta = 0.9
adagrad_eps = 1e-8
rmsprop_beta = 0.9
rmsprop_eps = 1e-8
adam_beta1 = 0.5
adam_beta2 = 0.9
adam_eps = 1e-8


