#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: Variables.py
@time: 2018/09/13 11:39
@description:
methods about Variables
"""
import sys
sys.path.append("..")
from SimpleDL.util.config import *

def Variable(shape,init_variables = init_variables_random):
    var = np.random.uniform(-5,5,shape)
    global_variables_list.extend(var.flatten().tolist())
    global_variables_shape.append(var.shape)
    return var





