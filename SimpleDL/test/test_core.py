#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_core.py
@time: 2018/09/14 21:34
@description:
test for core.py
"""
import sys
sys.path.append("..")
from SimpleDL.util.core import *

class TestCore:

    def test_conv2d(self):
        x = np.random.random((200,28,28,3)) # (num,width,height,channels)
        w = np.random.random((5,3,3,3)) #(output_dim,kernel_width,kernel_height,input_dim)
        kernel_size = (3,3)
        input_dim = 3
        output_dim = 5
        step_size = 3
        padding = 'valid'
        conv_output = conv2d(x,w,kernel_size,input_dim,output_dim,step_size,padding)
        print(conv_output.shape)

if __name__ == '__main__':
    test = TestCore()
    test.test_conv2d()

