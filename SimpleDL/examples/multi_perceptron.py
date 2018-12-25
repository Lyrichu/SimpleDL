#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: multi_perceptron.py
@time: 2018/09/13 12:07
@description:
multi perceptron examples
"""
from warnings import warn
from SimpleDL.examples.data import generate_data_sin_cos_exp
from SimpleDL.util.config import *
from SimpleDL.util.Variables import Variable
from SimpleDL.util.functions import *
from SimpleDL.util.config import *
from SimpleDL.util.core import *
from SimpleDL.util.optimizers import *

class MultiPerceptron:
    def __init__(self,data_nums = 500,train_percent = 0.7,lr = lr,lr_schedule = lr_schedule,use_lr_schedule = True,
                 update_lr_per_epochs = 100,print_per_epochs = 100,
                 lr_init = lr_init,lr_schedule_epochs = lr_schedule_epochs,epochs = 500,
                 hidden_units = (10,5,1),activation_func = sigmoid
                 ):
        self.vars = None
        self.datas_num = data_nums
        self.input_shape = 2
        self.data = generate_data_sin_cos_exp(self.datas_num)
        self.train_percent = train_percent
        self.train_data = self.data[:,:int(self.datas_num*self.train_percent)]
        self.test_data = self.data[:,int(self.datas_num*self.train_percent):]
        self.train_x = self.train_data[:self.input_shape,:]
        self.train_y = self.train_data[self.input_shape,:]
        self.test_x = self.test_data[:self.input_shape, :]
        self.test_y = self.test_data[self.input_shape, :]
        self.lr = lr
        self.use_lr_schedule = use_lr_schedule
        self.print_per_epochs = print_per_epochs
        self.update_lr_per_epochs = update_lr_per_epochs
        self.lr_schedule = lr_schedule
        self.lr_init = lr_init
        self.lr_schedule_epochs = lr_schedule_epochs
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.activation_func = activation_func


    def train(self):
        W1 = Variable((self.hidden_units[0],self.input_shape))
        b1 = Variable((self.hidden_units[0],1))
        W2 = Variable((self.hidden_units[1],self.hidden_units[0]))
        b2 = Variable((self.hidden_units[1],1))
        W3 = Variable((self.hidden_units[2],self.hidden_units[1]))
        b3 = Variable((self.hidden_units[2],1))
        for i in range(1,self.epochs+1):
            if self.use_lr_schedule:
                if i % self.update_lr_per_epochs == 0:
                    self.lr_init *= self.lr_schedule
                self.vars = gradient_descent(self.train_x,self.train_y,self.target_func,self.loss_func,lr =self.lr_init)
            else:
                self.vars = gradient_descent(self.train_x, self.train_y, self.target_func, self.loss_func, lr=self.lr)
            y1 = self.target_func(self.train_x,*self.vars)
            loss = self.loss_func(y1,self.train_y)
            if i % self.print_per_epochs == 0:
                print("%d/%d,loss = %.3f" %(i,self.epochs,loss))


    def predict(self):
        if self.vars is None:
            warn("You should train first before predict!")
            exit(-1)
        y1 = self.target_func(self.test_x,*self.vars)
        predict_loss = self.loss_func(y1,self.test_y)
        print("loss on test data is:%.3f!" % predict_loss)



    def target_func(self,x,*args):
        W1,b1,W2,b2,W3,b3 = args
        y1 = self.activation_func(W1.dot(x)+b1)
        y2 = self.activation_func(W2.dot(y1)+b2)
        output = W3.dot(y2)+b3
        return output

    def loss_func(self,y1,y):
        return np.sqrt(np.square(y-y1).mean())






