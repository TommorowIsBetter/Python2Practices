#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/5/27 14:25
"""
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import *
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 构建数据集
data = datasets.load_iris()
X, y = data.data, data.target

x = preprocessing.StandardScaler().fit_transform(X)
dataset = ClassificationDataSet(x.shape[1], 1, nb_classes=3)
for i in range(len(y)):
    dataset.addSample(list(x[i]), list([y[i]]))


# 划分数据集
dataset._convertToOneOfMany()
dataTrain, dataTest = dataset.splitWithProportion(proportion=0.8)
x_train, y_train = dataTrain['input'], dataTrain['target']
x_test, y_test = dataTest['input'], dataTest['target']
print('Input dim:{} Output dim:{}'.format(dataTrain.indim, dataTrain.outdim))
print('Train: x = {} y = {}'.format(x_train.shape, y_train.shape))
print('Test: x = {} y = {}'.format(x_test.shape, y_test.shape))

# 训练网络
net = buildNetwork(dataTrain.indim, 5, dataTrain.outdim, outclass=SoftmaxLayer)
model = BackpropTrainer(net, dataTrain, learningrate=0.01, momentum=0.1, verbose=False)
model.trainUntilConvergence(maxEpochs=100)

predict_train = np.argmax(net.activateOnDataset(dataTrain), axis=1)
actual_train = np.argmax(y_train, axis=1)
train_acc = accuracy_score(actual_train, predict_train)

predict_test = np.argmax(net.activateOnDataset(dataTest), axis=1)
actual_test = np.argmax(y_test, axis=1)
test_acc = accuracy_score(actual_test, predict_test)

print('Train acc = ', round(train_acc, 2), ' Test acc = ', round(test_acc, 2))