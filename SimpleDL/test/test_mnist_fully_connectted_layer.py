#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_mnist_fully_connectted_layer.py
@time: 2018/09/13 18:11
@description:
test for mnist_fully_connected_layer.py
"""
import sys
sys.path.append("..")
from SimpleDL.util.functions import *
from SimpleDL.util.optimizers import *
from SimpleDL.examples.mnist_fully_connectted_layer import MNIST_FULLY_CONNETCTED_LAYER

class Test_MNIST_FULLY_CONNETCTED_LAYER:
    def __init__(self):
        self.mnist = MNIST_FULLY_CONNETCTED_LAYER(activation_func=sigmoid,lr_init=1,lr = 0.01,lr_schedule_epochs=5,
                                                  print_per_epochs=1,mini_batch=200,optimizer=rmsprop,
                                                epochs=100,use_lr_schedule=False)

    def test_train(self):
        self.mnist.train()

    def test_predict(self):
        self.mnist.predict()




if __name__ == '__main__':
    test = Test_MNIST_FULLY_CONNETCTED_LAYER()
    test.test_train()
    test.test_predict()
