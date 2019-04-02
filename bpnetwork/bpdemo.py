#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/4/2 19:11
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


# 获取训练结果(这里是平均误差)
def get_performance(output, target):
    summary = 0.0
    for i in xrange(len(output)):
        summary += abs(output[i] - target[i])
    return summary / len(output)


# 返回两个输入层，3个隐藏层，1个输出层的神经网络
net = buildNetwork(2, 3, 1)
print net.activate([2, 1])
# 产生一个数据集，这个数据集支持两维输入和一维输出
ds = SupervisedDataSet(2, 1)
# 添加相应的训练数据集
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (2,))
ds.addSample((2, 2), (4,))

ds_test = SupervisedDataSet(2, 1)
# 添加相应的测试数据集
ds_test.addSample((0, 0), (0,))
ds_test.addSample((2, 1), (3,))
# 打印数据集的大小
print len(ds)
# 通过标准的方式迭代输出
for inpt, target in ds:
    print inpt, target
# 访问input
print ds['input']
# 访问target
print ds['target']
trainer = BackpropTrainer(net, ds)
# 训练一千次数
trainer.trainEpochs(1000)
output = net.activateOnDataset(ds_test)
# 下面为预测输出的结果
print output
# 输出预测的误差
print get_performance(output, ds_test['target'])