#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_simple_binary_classification.py
@time: 2018/09/13 15:56
@description:
test for simple_binary_classfication
"""

import sys
sys.path.append("..")
from SimpleDL.util.functions import *
from SimpleDL.util.optimizers import *
from SimpleDL.examples.simple_binary_classification import SimpleBinaryClassification

class TestSimpleBinaryClassification:
    def __init__(self):
        self.simple_binary_classification = SimpleBinaryClassification(
            activation_func=sigmoid,lr_init=2.,
            lr = 0.005,print_per_epochs=100,optimizer=momentum,
            epochs=2000,use_lr_schedule=False)

    def test_train(self):
        self.simple_binary_classification.train()

    def test_predict(self):
        self.simple_binary_classification.predict()




if __name__ == '__main__':
    test = TestSimpleBinaryClassification()
    test.test_train()
    test.test_predict()

