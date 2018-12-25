#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_multi_perceptron.py
@time: 2018/09/13 13:45
@description:
test for multi_perceptron
"""
import sys
sys.path.append("..")
from SimpleDL.util.functions import *
from SimpleDL.examples.multi_perceptron import MultiPerceptron

class TestMultiPerceptron:
    def __init__(self):
        self.multi_perceptron = MultiPerceptron(activation_func=tanh,lr_init=2.,lr = 0.005,
                                                epochs=1000,use_lr_schedule=False)

    def test_train(self):
        self.multi_perceptron.train()

    def test_predict(self):
        self.multi_perceptron.predict()




if __name__ == '__main__':
    test = TestMultiPerceptron()
    test.test_train()
    test.test_predict()

