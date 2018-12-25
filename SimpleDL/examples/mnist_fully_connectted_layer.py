#!/usr/bin/env python3
# encoding: utf-8
"""
@version: 0.1
@author: lyrichu
@license: Apache Licence 
@contact: 919987476@qq.com
@site: http://www.github.com/Lyrichu
@file: mnist_fully_connectted_layer.py
@time: 2018/09/13 17:13
@description:
fully connected layer for mnist
"""

from warnings import warn
from SimpleDL.examples.data import load_mnist
from SimpleDL.util.Variables import Variable
from SimpleDL.util.functions import *
from SimpleDL.util.config import *
from SimpleDL.util.core import *
from SimpleDL.util.optimizers import *

class MNIST_FULLY_CONNETCTED_LAYER:
    def __init__(self,lr = lr,lr_schedule = lr_schedule,use_lr_schedule = True,
                 update_lr_per_epochs = 100,print_per_epochs = 100,
                 lr_init = lr_init,lr_schedule_epochs = lr_schedule_epochs,epochs = 500,mini_batch = 10,
                 hidden_units = (4,5,10),activation_func = sigmoid,optimizer = gradient_descent
                 ):
        self.vars = None
        self.train_x,self.test_x,self.train_y,self.test_y = load_mnist()
        self.train_x, self.test_x = self.train_x/255.,self.test_x/255.
        self.train_x = self.train_x.reshape((self.train_x.shape[0],-1)).T
        self.input_shape = self.train_x.shape[0]
        self.test_x = self.test_x.reshape((self.test_x.shape[0],-1)).T
        self.train_y = onehot_encoder(self.train_y,10)
        self.test_y = onehot_encoder(self.test_y,10)
        self.lr = lr
        self.use_lr_schedule = use_lr_schedule
        self.print_per_epochs = print_per_epochs
        self.update_lr_per_epochs = update_lr_per_epochs
        self.lr_schedule = lr_schedule
        self.lr_init = lr_init
        self.lr_schedule_epochs = lr_schedule_epochs
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.mini_batch = mini_batch
        self.activation_func = activation_func
        self.optimizer = optimizer

    def train(self):
        W1 = Variable((self.hidden_units[0],self.input_shape))
        b1 = Variable((self.hidden_units[0],1))
        W2 = Variable((self.hidden_units[1],self.hidden_units[0]))
        b2 = Variable((self.hidden_units[1],1))
        W3 = Variable((self.hidden_units[2], self.hidden_units[1]))
        b3 = Variable((self.hidden_units[2], 1))
        batches = self.train_x.shape[1]//self.mini_batch
        for i in range(1,self.epochs+1):
            for j in range(batches):
                mini_batch_x = self.train_x[:, j * self.mini_batch:(j + 1) * self.mini_batch]
                mini_batch_y = self.train_y[:, j * self.mini_batch:(j + 1) * self.mini_batch]
                if self.use_lr_schedule:
                    if i % self.update_lr_per_epochs == 0:
                        self.lr_init *= self.lr_schedule
                    if self.optimizer is adam:
                        self.vars = adam(mini_batch_x, mini_batch_y, self.target_func, self.loss_func,
                                                     lr=self.lr_init,epoch=i-1)
                    else:
                        self.vars = self.optimizer(mini_batch_x,mini_batch_y,self.target_func,self.loss_func,lr =self.lr_init)
                else:
                    if self.optimizer is adam:
                        self.vars = adam(mini_batch_x, mini_batch_y, self.target_func, self.loss_func,
                                         lr=self.lr, epoch=i - 1)
                    else:
                        self.vars = self.optimizer(mini_batch_x, mini_batch_y, self.target_func, self.loss_func,
                                                   lr=self.lr)
                y1 = self.target_func(self.train_x,*self.vars)
                loss = self.loss_func(y1,self.train_y)
                acc = self.accuracy(y1,self.train_y)
                if i % self.print_per_epochs == 0:
                    print("%d/%d,%d/%dloss = %.3f,accuracy:%.3f" %(i,self.epochs,j+1,batches,loss,acc))


    def predict(self):
        if self.vars is None:
            warn("You should train first before predict!")
            exit(-1)
        y1 = self.target_func(self.test_x,*self.vars)
        predict_loss = self.loss_func(y1,self.test_y)
        predict_acc = self.accuracy(y1,self.test_y)
        print("loss on test data is:%.3f,accuracy is:%.3f!" % (predict_loss,predict_acc))



    def target_func(self,x,*args):
        W1,b1,W2,b2,W3,b3 = args
        y1 = self.activation_func(W1.dot(x)+b1)
        y2 = self.activation_func(W2.dot(y1)+b2)
        output = softmax(W3.dot(y2)+b3)
        return output




    def loss_func(self,y1,y):
        '''
        the binary_classification loss,
        the cross entropy loss
        :param y1:
        :param y:
        :return: the cross entropy loss
        '''

        return -np.sum(np.sum(y*np.log(y1),axis=0).flatten())

    def accuracy(self,y1,y):
        return ((np.argmax(y,axis=0) == np.argmax(y1,axis=0)).astype(np.float32)).sum()/y.shape[1]


