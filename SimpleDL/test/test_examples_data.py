#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: test_examples_data.py
@time: 2018/09/13 16:56
@description:
test for examples/data.py
"""
import sys
sys.path.append("..")
from warnings import warn
from SimpleDL.examples.data import *

class TestExampleData:
    def assertTrue(self,x):
        if x:
            print("assert True!")
        else:
            warn("assert False!")
            exit(-1)

    def test_generate_data_sin_cos_exp(self):
        self.assertTrue(generate_data_sin_cos_exp(100).shape == (3,100))

    def test_generate_simple_binary_data(self):
        self.assertTrue(generate_simple_binary_data(100).shape == (3,100))

    def test_load_mnist(self):
        x_train, x_test, y_train, y_test = load_mnist()
        self.assertTrue(x_train.shape == (60000,28,28))



if __name__ == '__main__':
    test = TestExampleData()
    test.test_generate_data_sin_cos_exp()
    test.test_generate_simple_binary_data()
    test.test_load_mnist()
