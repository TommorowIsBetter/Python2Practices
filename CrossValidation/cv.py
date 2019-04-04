#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/4/4 16:36
"""


from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.validation import CrossValidator, Validator, ModuleValidator


# 返回两个输入层，3个隐藏层，1个输出层的神经网络
net = buildNetwork(2, 3, 1)
# 产生一个数据集，这个数据集支持两维输入和一维输出
ds = SupervisedDataSet(2, 1)
# 添加相应的训练数据集
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (2,))
ds.addSample((2, 2), (4,))
ds.addSample((3, 3), (6,))
ds.addSample((2, 3), (5,))
ds.addSample((5, 5), (10,))
trainer = BackpropTrainer(net, ds)
# 训练一千次数
trainer.trainEpochs(2000)
modval = ModuleValidator()
print("the result of cv.")
# modval.MSE均方误差，4折交叉验证，ds为全部数据集既包括训练集和测试集
cv = CrossValidator(trainer, ds, n_folds=4, valfunc=modval.MSE)
print cv.validate()
