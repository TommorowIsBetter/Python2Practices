#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:Wang Yan
@ide:PyCharm
@time:2019/4/2 20:17
"""
# 把网络模型进行保存下来，后边需要的时候可以读取
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader

net = buildNetwork(2, 4, 1)
NetworkWriter.writeToFile(net, 'testNet.xml')
net = NetworkReader.readFrom('testNet.xml')